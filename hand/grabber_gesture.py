# grabber_gesture.py
# -*- coding: utf-8 -*-
"""
GestureGrabber: RealSense D435i + MediaPipe Hands
- Detect a hand and compute 3D metrics between thumb tip (4) and index tip (8).
- Outputs: XYZ of both tips (meters), delta_xyz, Euclidean distance (meters),
  and 3D angle between last thumb and index segments (vectors 4-3 and 8-7).
- Depth is aligned to color, and deprojection uses color intrinsics.

Run directly:
    python grabber_gesture.py

Use as a library:
    from grabber_gesture import GestureGrabber
    
    # For higher resolution (1280x720):
    g = GestureGrabber(color_size=(1280, 720), depth_size=(1280, 720))
    g.start()
    
    # Get camera intrinsic matrix
    K = g.get_intrinsic_matrix()
    distCoeffs = g.get_distortion_coeffs()
    print("Camera matrix:", K)
    
    try:
        while True:
            frame_bgr, metrics = g.process_once(draw=True)
            if frame_bgr is None: 
                continue
            # Do something with 'metrics'
    finally:
        g.stop()
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs


# ---------------------------
# Data structures
# ---------------------------

@dataclass
class PinchMetrics:
    """Container for one-frame pinch/angle measurements."""
    valid: bool
    thumb_tip_xyz_m: Optional[np.ndarray] = None   # (3,) meters
    index_tip_xyz_m: Optional[np.ndarray] = None   # (3,) meters
    delta_xyz_m: Optional[np.ndarray] = None       # (3,) meters (index - thumb)
    distance_m: Optional[float] = None             # Euclidean distance (meters)
    angle_deg: Optional[float] = None              # Angle between (4-3) and (8-7) in 3D
    palm_base_xyz_m: Optional[np.ndarray] = None
    palm_base_uv: Optional[Tuple[int, int]] = None
    thumb_tip_uv: Optional[Tuple[int, int]] = None # pixel coords on color image
    index_tip_uv: Optional[Tuple[int, int]] = None # pixel coords on color image
    fps: Optional[float] = None                    # processing FPS estimate

# ---------------------------
# Core class
# ---------------------------

class GestureGrabber:
    """
    A small utility to:
      - Start RealSense streams and align depth to color
      - Run MediaPipe Hands on color frames
      - Compute 3D metrics (XYZ, Δxyz, Euclidean distance) and 3D angle between last segments
      - Optionally draw overlays

    Parameters
    ----------
    color_size : tuple
        (width, height) for color stream. Default (640, 480).
    depth_size : tuple
        (width, height) for depth stream. Default (640, 480).
    fps : int
        Stream FPS. Default 30.
    max_num_hands : int
        MediaPipe max number of hands. Default 1.
    det_conf : float
        MediaPipe min detection confidence. Default 0.5.
    track_conf : float
        MediaPipe min tracking confidence. Default 0.5.
    median_win : int
        Window size for robust median depth sampling (odd). Default 5.
    ema_alpha : float
        Exponential moving average factor for smoothing 3D points and angle (0<alpha<=1).
        Use 1.0 to disable smoothing. Default 0.6.
    draw : bool
        Whether to draw overlay in process_once() when draw=True. Default True.
    """

    # MediaPipe landmark indices
    THUMB_TIP = 4
    THUMB_IP  = 3
    INDEX_TIP = 8
    INDEX_DIP = 7
    PALM_BASE = 0

    def __init__(
        self,
        color_size: Tuple[int, int] = (640, 480),
        depth_size: Tuple[int, int] = (640, 480),
        fps: int = 30,
        max_num_hands: int = 1,
        det_conf: float = 0.5,
        track_conf: float = 0.5,
        median_win: int = 5,
        ema_alpha: float = 0.6,
    ) -> None:

        self.color_w, self.color_h = color_size
        self.depth_w, self.depth_h = depth_size
        self.fps = fps
        self.median_win = median_win if median_win % 2 == 1 else median_win + 1
        self.ema_alpha = float(ema_alpha)

        # RealSense handles
        self.pipeline = None
        self.align = None
        self.color_intrin = None
        self.depth_scale = None

        # MediaPipe
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

        # State for smoothing
        self._ema_thumb = None  # np.ndarray (3,)
        self._ema_index = None  # np.ndarray (3,)
        self._ema_angle_state = None  # float

        self._t_prev = time.time()

        self.K = None                 # 3x3 camera matrix
        self.distCoeffs = None        # (N,) distortion
        self.R_cb = None              # board->camera rotation (3x3)
        self.t_cb = None              # board->camera translation (3,)
        self.R_bc = None              # camera->board rotation (3x3)
        self.t_bc = None              # camera->board translation (3,)
        self.world_valid = False

    # ---------- Public API ----------

    def start(self) -> None:
        """Start RealSense pipeline and prepare alignment and intrinsics."""
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.depth_w, self.depth_h, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.bgr8, self.fps)
        
        # Start pipeline and get intrinsics
        profile = self.pipeline.start(cfg)

        # Depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        # Align depth->color
        self.align = rs.align(rs.stream.color)

        # 添加内参标定 - 先抓一帧拿到内参，后面循环就不用每帧都算了
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        if color_frame:
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            intr_matrix = np.array([[intr.fx, 0, intr.ppx],
                                   [0, intr.fy, intr.ppy],
                                   [0, 0, 1]], dtype=np.float32)
            intr_coeffs = np.asarray(intr.coeffs)
            
            # 更新内参矩阵
            self.K = intr_matrix.astype(np.float64)
            self.distCoeffs = intr_coeffs.astype(np.float64)[:5] if len(intr_coeffs) >= 5 else np.zeros(5, dtype=np.float64)
            
            print("[GestureGrabber] Camera Intrinsic Matrix:")
            print(intr_matrix)
            print(f"[GestureGrabber] Distortion coefficients: {intr_coeffs}")
        
        # Color intrinsics for deprojection (保持兼容性)
        color_stream = profile.get_stream(rs.stream.color)
        color_vs_profile = rs.video_stream_profile(color_stream)
        self.color_intrin = color_vs_profile.get_intrinsics()

        print(f"[GestureGrabber] Started. Depth scale = {self.depth_scale:.6f} m/unit.")
        print(f"[GestureGrabber] Color intrinsics: fx={self.color_intrin.fx:.2f}, "
              f"fy={self.color_intrin.fy:.2f}, cx={self.color_intrin.ppx:.2f}, cy={self.color_intrin.ppy:.2f}")

    def stop(self) -> None:
        """Release resources."""
        try:
            if self.hands is not None:
                self.hands.close()
        finally:
            self.hands = None
            if self.pipeline is not None:
                self.pipeline.stop()
            self.pipeline = None
            cv2.destroyAllWindows()
            print("[GestureGrabber] Stopped.")

    def get_intrinsic_matrix(self) -> Optional[np.ndarray]:
        """
        获取相机内参矩阵
        
        Returns
        -------
        K : np.ndarray or None
            3x3 相机内参矩阵，如果还未初始化则返回 None
        """
        return self.K.copy() if self.K is not None else None
    
    def get_distortion_coeffs(self) -> Optional[np.ndarray]:
        """
        获取相机畸变系数
        
        Returns
        -------
        distCoeffs : np.ndarray or None
            畸变系数数组，如果还未初始化则返回 None
        """
        return self.distCoeffs.copy() if self.distCoeffs is not None else None

    def process_once(self, draw: bool = True) -> Tuple[Optional[np.ndarray], PinchMetrics]:
        """
        Grab one aligned frame, run hand detection, compute metrics, and (optionally) draw.

        Returns
        -------
        frame_bgr : np.ndarray or None
            Color frame with overlay if draw=True, else raw color frame. None if frames missing.
        metrics : PinchMetrics
            One-frame measurements (valid flag indicates success).
        """
        # Fetch frames
        if self.pipeline is None:
            return None, PinchMetrics(valid=False)

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, PinchMetrics(valid=False)

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())  # uint16 units
        depth_m = depth_raw * self.depth_scale             # float meters

        h, w, _ = color_bgr.shape
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        thumb_tip_xyz = None
        index_tip_xyz = None
        delta_xyz = None
        distance = None
        angle_deg = None
        u_ttip = v_ttip = u_itip = v_itip = None

        if results.multi_hand_landmarks:
            # Use the first detected hand by default
            lms = results.multi_hand_landmarks[0].landmark

            def lm2uv(lm):
                u = int(round(lm.x * w))
                v = int(round(lm.y * h))
                return self._clamp(u, 0, w - 1), self._clamp(v, 0, h - 1)

            # Pixels for the four needed landmarks
            u_ttip, v_ttip = lm2uv(lms[self.THUMB_TIP])
            u_tprev, v_tprev = lm2uv(lms[self.THUMB_IP])
            u_itip, v_itip = lm2uv(lms[self.INDEX_TIP])
            u_iprev, v_iprev = lm2uv(lms[self.INDEX_DIP])

            u_base, v_base = lm2uv(lms[self.PALM_BASE])
            u_ttip, v_ttip = lm2uv(lms[self.THUMB_TIP])
            u_itip, v_itip = lm2uv(lms[self.INDEX_TIP])

            z_base = self._median_depth(depth_m, u_base, v_base, self.median_win)
            z_ttip = self._median_depth(depth_m, u_ttip, v_ttip, self.median_win)
            z_itip = self._median_depth(depth_m, u_itip, v_itip, self.median_win)

            if z_base > 0 and z_ttip > 0 and z_itip > 0:
                base_xyz  = self._deproject(u_base,  v_base,  z_base)
                thumb_xyz = self._deproject(u_ttip,  v_ttip,  z_ttip)
                index_xyz = self._deproject(u_itip,  v_itip,  z_itip)

                # 点位平滑
                thumb_xyz = self._ema_vec("thumb", thumb_xyz)
                index_xyz = self._ema_vec("index", index_xyz)

                # Δxyz 与距离（仍按指尖对指尖）
                delta_xyz = index_xyz - thumb_xyz
                distance  = float(np.linalg.norm(delta_xyz))

                # 新角度：以“掌根点”为顶点，向量 base->tip
                v_bt = thumb_xyz - base_xyz
                v_bi = index_xyz - base_xyz
                angle_deg = self._angle_between(v_bt, v_bi)
                angle_deg = self._ema_angle(angle_deg)

                # 记录到 metrics
                thumb_tip_xyz = thumb_xyz
                index_tip_xyz = index_xyz
                u_ttip, v_ttip = u_ttip, v_ttip
                u_itip, v_itip = u_itip, v_itip
                # 还要把 base 写进 metrics（若你按上一步加了字段）
                palm_base_xyz = base_xyz
                palm_base_uv  = (u_base, v_base)


        # FPS
        now = time.time()
        fps = 1.0 / (now - self._t_prev) if now > self._t_prev else 0.0
        self._t_prev = now

        metrics = PinchMetrics(
            valid=thumb_tip_xyz is not None and index_tip_xyz is not None,
            thumb_tip_xyz_m=thumb_tip_xyz,
            index_tip_xyz_m=index_tip_xyz,
            delta_xyz_m=delta_xyz,
            distance_m=distance,
            angle_deg=angle_deg,
            thumb_tip_uv=(u_ttip, v_ttip) if u_ttip is not None else None,
            index_tip_uv=(u_itip, v_itip) if u_itip is not None else None,
            fps=fps,
        )

        # Draw
        out = color_bgr
        if draw and metrics.valid:
            out = self._draw_overlay(
                color_bgr,
                metrics,
                # also pass prev joints for short segments on 2D
                (u_tprev, v_tprev),
                (u_iprev, v_iprev),
            )
        elif draw:
            out = color_bgr.copy()
            cv2.putText(out, "Invalid depth or no hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # FPS text
        if out is not None:
            cv2.putText(out, f"FPS: {fps:.1f}", (10, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)

        return out, metrics

    # ---------- Internals ----------

    @staticmethod
    def _clamp(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> Optional[float]:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < eps or nb < eps:
            return None
        cosang = float(np.dot(a, b) / (na * nb))
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    def _median_depth(self, depth_m: np.ndarray, u: int, v: int, w: int) -> float:
        h, W = depth_m.shape
        half = w // 2
        u0, u1 = self._clamp(u - half, 0, W - 1), self._clamp(u + half + 1, 0, W)
        v0, v1 = self._clamp(v - half, 0, h - 1), self._clamp(v + half + 1, 0, h)
        window = depth_m[v0:v1, u0:u1].reshape(-1)
        window = window[window > 0.0]
        if window.size == 0:
            return 0.0
        return float(np.median(window))

    def _deproject(self, u: int, v: int, z_m: float) -> np.ndarray:
        pt = rs.rs2_deproject_pixel_to_point(self.color_intrin, [float(u), float(v)], float(z_m))
        return np.array(pt, dtype=np.float32)  # (X,Y,Z) in meters (color camera frame)

    def _ema_vec(self, which: str, x: np.ndarray) -> np.ndarray:
        """EMA smoothing for 3D points."""
        a = self.ema_alpha
        if a >= 0.999:  # effectively off
            return x
        if which == "thumb":
            if self._ema_thumb is None:
                self._ema_thumb = x.copy()
            self._ema_thumb = (1 - a) * self._ema_thumb + a * x
            return self._ema_thumb
        else:
            if self._ema_index is None:
                self._ema_index = x.copy()
            self._ema_index = (1 - a) * self._ema_index + a * x
            return self._ema_index

    def _ema_angle(self, ang: Optional[float]) -> Optional[float]:
        if ang is None:
            return None
        a = self.ema_alpha
        if a >= 0.999:
            return ang
        if self._ema_angle_state is None:
            self._ema_angle_state = float(ang)
        self._ema_angle_state = (1 - a) * self._ema_angle_state + a * float(ang)
        return self._ema_angle_state
    
        # ---------- Extrinsic: chessboard as world/desk frame ----------
    def calibrate_board(self, pattern_cols: int, pattern_rows: int, square_size_m: float = 0.015,
                        draw_preview: bool = True) -> bool:
        """
        更稳的单帧外参标定：优先用 findChessboardCornersSB（带 EXHAUSTIVE/ACCURACY），
        失败再回退到经典 findChessboardCorners。给出重投影 RMSE 作为质量指标。
        """
        if self.pipeline is None or self.K is None or self.distCoeffs is None:
            print("[Calib] Pipeline or intrinsics not ready.")
            return False

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            print("[Calib] No color frame.")
            return False

        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # 预处理：轻微去噪 + 增强对比度，压制背景网格干扰
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        pattern_size = (pattern_cols, pattern_rows)
        corners = None
        ok = False

        # 1) 优先 SB 版（OpenCV ≥ 4.5 有）
        if hasattr(cv2, "findChessboardCornersSB"):
            flags_sb = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags_sb)

        # 2) 回退到经典法
        if not ok:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if not ok or corners is None:
            print("[Calib] Chessboard NOT found. Try closer view / reduce glare / fill more FOV.")
            if draw_preview:
                cv2.imshow("Calib Preview", color); cv2.waitKey(1)
            return False

        # 亚像素优化
        if corners.dtype != np.float32:
            corners = corners.astype(np.float32)
        cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        )

        # 3D 目标点（棋盘坐标系，Z=0）
        objp = np.zeros((pattern_rows * pattern_cols, 3), np.float64)
        grid = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2).astype(np.float64)
        objp[:, :2] = grid * float(square_size_m)

        # PnP：board->camera
        ok, rvec, tvec = cv2.solvePnP(objp, corners, self.K, self.distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            print("[Calib] solvePnP failed.")
            return False

        R_cb, _ = cv2.Rodrigues(rvec); t_cb = tvec.reshape(3)
        self.R_cb, self.t_cb = R_cb, t_cb
        self.R_bc = R_cb.T
        self.t_bc = -self.R_bc @ t_cb
        self.world_valid = True

        # 质量：重投影误差
        reproj, _ = cv2.projectPoints(objp, rvec, tvec, self.K, self.distCoeffs)
        reproj = reproj.reshape(-1, 2); obs = corners.reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((reproj - obs) ** 2, axis=1))))
        print(f"[Calib] OK. Reproj RMSE = {rmse:.2f} px")

        if draw_preview:
            vis = color.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            # 坐标轴（从第一个角点出发）
            origin = corners[0].ravel()
            axis_len = square_size_m * 3.0
            axes = np.float64([[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]])
            imgpts, _ = cv2.projectPoints(axes, rvec, tvec, self.K, self.distCoeffs)
            imgpts = imgpts.reshape(-1, 2).astype(int)
            o = tuple(origin.astype(int))
            cv2.line(vis, o, tuple(imgpts[0]), (0, 0, 255), 3) # X-red
            cv2.line(vis, o, tuple(imgpts[1]), (0, 255, 0), 3) # Y-green
            cv2.line(vis, o, tuple(imgpts[2]), (255, 0, 0), 3) # Z-blue
            cv2.putText(vis, f"RMSE={rmse:.2f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,0), 2)
            cv2.imshow("Calib Preview", vis); cv2.waitKey(300)

        return True

    def cam_to_chessboard(self, Xc: np.ndarray) -> np.ndarray:
        """
        Transform 3D points from camera frame to board/world frame.
        Xc: (3,) or (N,3)
        """
        if not self.world_valid:
            raise RuntimeError("World pose not set. Call calibrate_board() first.")
        Xc = np.asarray(Xc, dtype=np.float64)
        if Xc.ndim == 1:
            return (self.R_bc @ Xc) + self.t_bc
        else:
            return (Xc @ self.R_bc.T) + self.t_bc  # (N,3)



    def _draw_overlay(
        self,
        img_bgr: np.ndarray,
        metrics: PinchMetrics,
        thumb_prev_uv: Tuple[int, int],
        index_prev_uv: Tuple[int, int],
    ) -> np.ndarray:
        """Draw landmarks-like lines and numeric texts for UX."""
        out = img_bgr.copy()

        # Short segments to tips
        if metrics.palm_base_uv and metrics.thumb_tip_uv:
            cv2.line(out, metrics.palm_base_uv, metrics.thumb_tip_uv, (0, 255, 255), 2)
        if metrics.palm_base_uv and metrics.index_tip_uv:
            cv2.line(out, metrics.palm_base_uv, metrics.index_tip_uv, (0, 255, 255), 2)

        if metrics.palm_base_uv:
            bx, by = metrics.palm_base_uv
            # 避免贴边，稍微偏移一下
            tx = min(max(bx + 10, 10), out.shape[1]-220)
            ty = min(max(by - 10, 30), out.shape[0]-10)
            ang_txt = "Angle(base->tips, 3D): "
            ang_txt += f"{metrics.angle_deg:.1f} deg" if metrics.angle_deg is not None else "N/A"
            cv2.putText(out, ang_txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2, cv2.LINE_AA)

        # Tiny circles on the tips
        if metrics.thumb_tip_uv:
            cv2.circle(out, metrics.thumb_tip_uv, 6, (0, 0, 255), 2)
        if metrics.index_tip_uv:
            cv2.circle(out, metrics.index_tip_uv, 6, (255, 0, 0), 2)

        # Texts (mm for readability)
        if metrics.delta_xyz_m is not None and metrics.distance_m is not None:
            dx, dy, dz = (metrics.delta_xyz_m * 1000.0).tolist()
            dist_mm = metrics.distance_m * 1000.0
            y0 = 30
            cv2.putText(out, f"Thumb-Index delta [mm]: dx={dx:+.1f}, dy={dy:+.1f}, dz={dz:+.1f}",
                        (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2, cv2.LINE_AA)
            cv2.putText(out, f"Euclidean distance: {dist_mm:.1f} mm",
                        (10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2, cv2.LINE_AA)

        ang_txt = "Angle(last segments, 3D): "
        ang_txt += f"{metrics.angle_deg:.1f} deg" if metrics.angle_deg is not None else "N/A"
        cv2.putText(out, ang_txt, (10, 30 + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2, cv2.LINE_AA)
        
        # Extra text in board/world frame if calibrated
        if self.world_valid and metrics.thumb_tip_xyz_m is not None and metrics.index_tip_xyz_m is not None:
            th_w = self.cam_to_chessboard(metrics.thumb_tip_xyz_m)
            ix_w = self.cam_to_chessboard(metrics.index_tip_xyz_m)
            d_w = (ix_w - th_w) * 1000.0  # mm
            dxw, dyw, dzw = d_w.tolist()
            y_add = 30 + 56 + 28  # below the previous lines
            cv2.putText(out, f"Thumb-Index delta (board) [mm]: dx={dxw:+.1f}, dy={dyw:+.1f}, dz={dzw:+.1f}",
                        (10, y_add), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2, cv2.LINE_AA)


        return out

    @staticmethod
    def _draw_arc(img, origin, dir_a, dir_b, length=60, steps=22):
        """Simple dotted arc between two 2D directions."""
        def nrm(d):
            n = float(np.linalg.norm(d[:2]))
            return d[:2] / (n + 1e-9)
        a = nrm(dir_a)
        b = nrm(dir_b)
        ang_a = math.atan2(a[1], a[0])
        ang_b = math.atan2(b[1], b[0])
        dtheta = (ang_b - ang_a + math.pi) % (2 * math.pi) - math.pi
        for i in range(steps + 1):
            t = ang_a + dtheta * (i / steps)
            p = (int(origin[0] + length * math.cos(t)),
                 int(origin[1] + length * math.sin(t)))
            cv2.circle(img, p, 1, (0, 255, 255), -1)


# ---------------------------
# Demo (optional)
# ---------------------------

def _demo():
    g = GestureGrabber(ema_alpha=0.6, median_win=5)
    g.start()
    
    # 显示获取到的内参矩阵
    intr_matrix = g.get_intrinsic_matrix()
    dist_coeffs = g.get_distortion_coeffs()
    if intr_matrix is not None:
        print("\n[Demo] Camera Intrinsic Matrix:")
        print(intr_matrix)
        print(f"[Demo] Distortion coefficients: {dist_coeffs}")
    
    try:
        while True:
            frame, m = g.process_once(draw=True)
            if frame is not None:
                cv2.imshow("GestureGrabber Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):  # calibrate with a 11x8 board (inner corners), 15 mm squares
                ok = g.calibrate_board(pattern_cols=11, pattern_rows=8, square_size_m=0.015)
                print(f"[Demo] Calibrate: {ok}")
            elif key == ord('r'):  # reset
                g.world_valid = False
                g.R_cb = g.t_cb = g.R_bc = g.t_bc = None
                print("[Demo] World pose cleared.")
            # Example: print numeric results
            if m.valid and (int(time.time() * 10) % 10 == 0):  # throttle prints
                print(
                    f"XYZ_th={m.thumb_tip_xyz_m}, XYZ_ix={m.index_tip_xyz_m}, "
                    f"Δxyz={m.delta_xyz_m}, dist={m.distance_m:.4f} m, angle={m.angle_deg}"
                )
    finally:
        g.stop()


if __name__ == "__main__":
    _demo()
