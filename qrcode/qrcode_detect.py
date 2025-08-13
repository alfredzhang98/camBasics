from CameraCalibration import *
import math, time
from pathlib import Path
import numpy as np
import cv2
import pyrealsense2 as rs

class QrcodeDetect:
    def __init__(self, id_list: list[int], qrcode_size_mm: float, aruco_dict_id: int, width: int, height: int, external_calibration_path: str = "t_T_cam.yml"):
        self.id_list = id_list
        self.qrcode_size_mm = qrcode_size_mm
        self.adict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        try:
            self.params = cv2.aruco.DetectorParameters()
        except Exception:
            self.params = cv2.aruco.DetectorParameters_create()
        self.detector = cv2.aruco.ArucoDetector(self.adict, self.params)

        # cam init
        self.cam = CameraInit(
            sensor_names=["RGB", "DEPTH"],
            width={"RGB": width, "DEPTH": width},
            height={"RGB": height, "DEPTH": height}
        )
        self._align = rs.align(rs.stream.color)
        self.K, self.dist, self.intr = CameraIntrinsicCalibrator.get_device_intrinsics(height=height, width=width)

        # Load external calibration
        self.t_T_cam = CameraExternalCalibrator.load_from_yaml(external_calibration_path)
        self.R_t_cam = self.t_T_cam[:3,:3]
        self.t_t_cam = self.t_T_cam[:3, 3:4]
        self._thread = None

        self._ema = {}           # id -> (3,1)
        self._last_seen = {}     # id -> 帧编号
        self._frame_idx = 0
        self._ema_alpha = 0.2   # 0.2~0.35 可调
        self._jump_mm = 25.0     # 单帧限幅

        self._tf = rs.temporal_filter()
        self._sf = rs.spatial_filter()
        self._hf = rs.hole_filling_filter()


    def _smooth_pos(self, mid, p_raw):
        prev = self._ema.get(mid)
        if prev is None:
            self._ema[mid] = p_raw.copy()
        else:
            delta = np.clip(p_raw - prev, -self._jump_mm, self._jump_mm)
            p_lim = prev + delta
            a = self._ema_alpha
            self._ema[mid] = a * p_lim + (1 - a) * prev
        self._last_seen[mid] = self._frame_idx
        return self._ema[mid]

    def _maybe_reset_ema(self, mid, miss_thresh=5):
        if mid in self._last_seen and (self._frame_idx - self._last_seen[mid]) > miss_thresh:
            self._ema.pop(mid, None)
            self._last_seen.pop(mid, None)

    def process_frame(self):

        self._frame_idx += 1

        frames = self.cam.get_frames()
        if frames is None or "RGB" not in frames or "DEPTH" not in frames:
            return None

        color_frame = frames["RGB"]
        depth_frame = frames["DEPTH"]
        depth_frame = self._sf.process(depth_frame)
        depth_frame = self._tf.process(depth_frame)
        depth_frame = self._hf.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()

        depth_intr = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

        img = self.cam.convert_frame_to_array(color_frame)
        if img is None:
            return None

        # Visualize: copy from original image
        vis = img.copy()

        # Detect ArUco markers very easy packaged by OpenCV
        corners, ids, _ = self.detector.detectMarkers(img)
        if ids is None or len(ids) == 0:
            cv2.putText(vis, "No markers", (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return vis

        # Filter out unwanted IDs
        ids = ids.flatten()
        keep_idx = [i for i, mid in enumerate(ids) if int(mid) in self.id_list]
        if len(keep_idx) == 0:
            cv2.putText(vis, f"IDs not in {self.id_list}", (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return vis

        # Keep only the corners and IDs we want
        corners_keep = [corners[i] for i in keep_idx]
        # id in column
        ids_keep     = ids[keep_idx].astype(np.int32).reshape(-1, 1)

        # The ids is available so we can estimate their pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners_keep, self.qrcode_size_mm, self.K, self.dist
        )
        # draw markers
        cv2.aruco.drawDetectedMarkers(vis, corners_keep, ids_keep)


        data = {}
        for j, mid in enumerate(ids_keep.flatten().tolist()):
            c4 = corners_keep[j].reshape(4, 2)
            center_x = int((c4[0][0] + c4[2][0]) / 2)
            center_y = int((c4[0][1] + c4[2][1]) / 2)

            # cv2.circle(vis, (center_x, center_y), 6, (0,255,0), -1)
            # # draw c4 circle
            # for i in range(len(c4)):
            #     cv2.circle(vis, tuple(c4[i].astype(int)), 3, (0,0,255), -1)
            
            # 深度反投影四角
            # pts_cam = []
            # pts_center = []
            # for (u, v) in c4:
            #     z_m = QrcodeDetect.get_depth_avg(depth_frame, u, v, win=7)
            #     if z_m <= 0:
            #         pts_cam = None; break
            #     pts_cam.append(QrcodeDetect.deproject_pixel(depth_intr, u, v, z_m))
            # pts_center.append(QrcodeDetect.deproject_pixel(depth_intr, center_x, center_y, QrcodeDetect.get_depth_avg(depth_frame, center_x, center_y, win=3)))
            # if pts_cam is None or len(pts_cam) != 4:
            #     continue
            # try:
            #     pts_cam = np.stack(pts_cam, 0)
            # except Exception:
            #     continue
            # p_cam  = pts_cam.mean(axis=0, keepdims=True).T
            # p_cam_center = pts_center[0].reshape(3,1).astype(np.float64)
            # p_cam_final = p_cam

            zs = []
            for (uu, vv) in [*c4, (center_x, center_y)]:
                z = QrcodeDetect.get_depth_avg(depth_frame, uu, vv, win=7)  # from win=3 -> 7
                if z > 0: 
                    zs.append(z)
            if len(zs) < 3:
                continue
            z_med = float(np.median(zs))
            p_cam_final = QrcodeDetect.deproject_pixel(depth_intr, center_x, center_y, z_med).reshape(3,1)

            # Convert rotation vector to rotation matrix
            R_cam_obj, _ = cv2.Rodrigues(rvecs[j].reshape(3,1))

            self._maybe_reset_ema(mid)

            # cam -> chessboard
            # center point location transformation
            p_t = self.R_t_cam @ p_cam_final + self.t_t_cam
            p_t_filt = self._smooth_pos(mid, p_t)
            xk, yk, zk = p_t_filt.reshape(3)

            # cam -> chessboard
            # marker rotation transformation
            R_fix = QrcodeDetect.Rx(180)
            R_t_obj = self.R_t_cam @ R_cam_obj
            R_t_obj = R_t_obj @ R_fix
            ypr = QrcodeDetect.rmat_to_euler_zyx(R_t_obj)

            ##################
            axis_len = self.qrcode_size_mm * 0.7
            # cv2.drawFrameAxes(img, self.K, self.dist, rvecs[j], tvecs[j], axis_len, thickness=2)
            QrcodeDetect.draw_axes_safe(vis, self.K, self.dist, rvecs[j], tvecs[j], axis_len, thickness=2)

            def _fmt(v): 
                return f"{v:+06.2f}"
            center_px = (int(center_x), int(center_y))
            cv2.circle(vis, center_px, 3, (0,0,255), -1)

            txt_t = f"ID{mid:02d} t:[{_fmt(xk)},{_fmt(yk)},{_fmt(zk)}]mm"
            txt_r = f"YPR(z,y,x):[{_fmt(ypr[0])},{_fmt(ypr[1])},{_fmt(ypr[2])}]deg"

            cv2.putText(vis, txt_t, (center_px[0]+10, center_px[1]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,220,50), 2)
            cv2.putText(vis, txt_r, (center_px[0]+10, center_px[1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,220,50), 2)

            data[int(mid)] = {"p_t": p_t_filt, "center_px": center_px}

        if len(self.id_list) >= 2:
            idA, idB = self.id_list[:2]
            if idA in data and idB in data:
                pA = data[idA]["p_t"].reshape(3); pB = data[idB]["p_t"].reshape(3)
                dvec = pA - pB
                d    = float(np.linalg.norm(dvec))
                d_xy = float(np.linalg.norm(dvec[:2]))
                d_z  = float(abs(dvec[2]))
                cA = data[idA]["center_px"]; cB = data[idB]["center_px"]
                cv2.line(vis, cA, cB, (0,255,255), 2)
                midpt = ((cA[0]+cB[0])//2, (cA[1]+cB[1])//2)
                cv2.putText(vis, f"d={d:.1f}mm  d_xy={d_xy:.1f}  dz={d_z:.1f}",
                            (midpt[0]+10, midpt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        return vis

    def start_threaded(self):
        """
        Start the marker detection in a separate thread.
        """
        import threading
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self._thread
    
    def run(self):
        self.cam.start_pipeline()
        try:
            while True:
                vis = self.process_frame()
                if vis is None:
                    cv2.waitKey(1)
                    continue

                cv2.imshow("Aruco Pose", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):   # ESC 或 q 退出
                    break
        finally:
            try:
                self.cam.stop_pipeline()
            except Exception:
                pass
            cv2.destroyAllWindows()


    @staticmethod
    def Rz(deg):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c,-s,0],
                         [ s, c,0],
                         [ 0, 0,1]], dtype=np.float64)

    @staticmethod
    def Ry(deg):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]], dtype=np.float64)

    @staticmethod
    def Rx(deg):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[ 1, 0, 0],
                         [ 0,  c,-s],
                         [ 0,  s, c]], dtype=np.float64)

    @staticmethod
    def draw_axes_safe(img, K, dist, rvec, tvec, length, thickness=3):
        if hasattr(cv2, "drawFrameAxes"):
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, length, thickness)
            return
        if hasattr(cv2, "aruco") and hasattr(cv2.aruco, "drawAxis"):
            cv2.aruco.drawAxis(img, K, dist, rvec, tvec, length)
            return
        # axis_obj = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
        # imgpts, _ = cv2.projectPoints(axis_obj, rvec, tvec, K, dist)
        # imgpts = imgpts.reshape(-1,2).astype(int)
        # o,x,y,z = map(tuple, imgpts)
        # cv2.line(img, o, x, (0,0,255), thickness)
        # cv2.line(img, o, y, (0,255,0), thickness)
        # cv2.line(img, o, z, (255,0,0), thickness)

    @staticmethod
    def rmat_to_euler_zyx(R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-9:
            yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
            pitch = math.degrees(math.atan2(-R[2,0], sy))
            roll  = math.degrees(math.atan2(R[2,1], R[2,2]))
        else:
            yaw   = math.degrees(math.atan2(-R[0,1], R[1,1]))
            pitch = math.degrees(math.atan2(-R[2,0], sy))
            roll  = 0.0
        return yaw, pitch, roll  # about (Z,Y,X)

    @staticmethod
    def get_depth_avg(depth_frame, u, v, win=3):
        u = int(round(u)); v = int(round(v))
        h = depth_frame.get_height(); w = depth_frame.get_width()
        u0 = max(0, u - win); u1 = min(w-1, u + win)
        v0 = max(0, v - win); v1 = min(h-1, v + win)
        zs = []
        for yy in range(v0, v1+1):
            for xx in range(u0, u1+1):
                z = depth_frame.get_distance(xx, yy)
                if z > 0: zs.append(z)
        if not zs:
            return depth_frame.get_distance(u, v)
        return float(np.median(zs))
    
    @staticmethod
    def get_depth_raw(depth_frame, u, v):
        z = depth_frame.get_distance(u, v)
        return float(z)

    @staticmethod
    def deproject_pixel(intr, u, v, z_m):
        # 返回相机坐标（mm）
        X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(z_m))
        return np.array([X, Y, Z], dtype=np.float64) * 1000.0  # m -> mm



if __name__ == "__main__":
    # init 
    qrcode = QrcodeDetect(id_list=[23,45], qrcode_size_mm=24.5, aruco_dict_id=cv2.aruco.DICT_4X4_50, width=1280, height=720)

    qrcode.start_threaded()

    # main loop
    try:
        while True:
            if not qrcode._thread.is_alive():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        