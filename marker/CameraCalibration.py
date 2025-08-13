from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import glob
import time
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List

class CameraInit:
    def __init__(self,
                 sensor_names: list[str],
                 width: dict[str, int],
                 height: dict[str, int] = None,
                 fps: dict[str, int] = None,
                 exposure: dict[str, int] = None,
                 enable_projection: bool = False):

        """
        Example usage:
        cam = CameraInit(sensor_names=["IR1", "IR2"], width={"IR1": 1280, "IR2": 1280}, height={"IR1": 720, "IR2": 720}, fps={"IR1": 30, "IR2": 30}, exposure={"IR1": 500, "IR2": 500})
        """

        if isinstance(sensor_names, (str, bytes)):
            sensor_names = [sensor_names]
        valid = {"RGB", "IR1", "IR2", "DEPTH"}
        self.sensor_names = [s.upper() for s in sensor_names]
        for s in self.sensor_names:
            if s not in valid:
                raise ValueError(f"Unsupported sensor '{s}'. Use {sorted(valid)}.")

        def get_param_dict(p_dict, default):
            return {name: p_dict.get(name, default) if isinstance(p_dict, dict) else default
                    for name in self.sensor_names}

        # If the parameter is not provided, use the default value 640 X 480 30fps
        self.width    = get_param_dict(width, 640)
        self.height   = get_param_dict(height, 480)
        self.fps      = get_param_dict(fps, 30)
        self.exposure = get_param_dict(exposure, None)

        # frame cache
        self.frame_cache = {"RGB": None, "IR1": None, "IR2": None, "DEPTH": None}

        ############################# User Configuration #############################
        # Stream Configuration Cheat-sheet:
        #   Stream     : Type of data stream (Depth, Color, Infrared)
        #   Index      : Sensor index (for IR, 0=left, 1=right; Depth/Color always 0)
        #   Resolution : Width×Height in pixels
        #   Format     : Pixel format (e.g., z16, rgb8, y8)
        #   FPS        : Supported frame rates (frames per second)
        #
        # ┌────────────┬───────┬────────────┬──────────┬──────────────────┐
        # │ Stream     │ Index │ Resolution │ Format   │ FPS              │
        # ├────────────┼───────┼────────────┼──────────┼──────────────────┤
        # │ Depth      │ 0     │ 640×480    │ z16      │ 6, 15, 30, 60    │
        # │ Depth      │ 0     │ 1280×720   │ z16      │ 15, 30           │
        # │ Color      │ 0     │ 640×480    │ rgb8/bgr8│ 6, 15, 30, 60    │
        # │ Color      │ 0     │ 1280×720   │ rgb8/bgr8│ 15, 30           │
        # │ Color      │ 0     │ 1920×1080  │ rgb8/bgr8│ 15, 30           │
        # │ Infrared   │ 0     │ 640×480    │ y8       │ 6, 15, 30        │
        # │ Infrared   │ 1     │ 640×480    │ y8       │ 6, 15, 30        │
        # │ Infrared   │ 0     │ 1280×720   │ y8       │ 6, 15, 30        │
        # │ Infrared   │ 1     │ 1280×720   │ y8       │ 6, 15, 30        │
        # └────────────┴───────┴────────────┴──────────┴──────────────────┘
        #
        allowed = {
            "DEPTH": {(640, 480): {6, 15, 30, 60}, (1280, 720): {15, 30}},
            "RGB": {(640, 480): {6, 15, 30, 60}, (1280, 720): {15, 30}, (1920, 1080): {15, 30}},
            "IR": {(640, 480): {6, 15, 30}, (1280, 720): {6, 15, 30}}
        }

        for name in self.sensor_names:
            key = "IR" if name in ("IR1", "IR2") else name
            wh = (self.width[name], self.height[name])
            fps = self.fps[name]
            assert wh in allowed[key] and fps in allowed[key][wh], \
                f"{name} not supported {wh[0]}x{wh[1]}@{fps} only allowed: {[(w,h,f) for (w,h),fs in allowed[key].items() for f in fs]}"


        self.ctx = rs.context()
        devs = self.ctx.query_devices()
        if not devs:
            raise RuntimeError("No RealSense devices found.")
        self.dev = devs[0]
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color) if "DEPTH" in self.sensor_names else None

        if "RGB" in self.sensor_names:
            self.config.enable_stream(rs.stream.color, 0,
                                      self.width["RGB"], self.height["RGB"],
                                      rs.format.bgr8, self.fps["RGB"])
        if "IR1" in self.sensor_names:
            self.config.enable_stream(rs.stream.infrared, 1,
                                      self.width["IR1"], self.height["IR1"],
                                      rs.format.y8, self.fps["IR1"])
        if "IR2" in self.sensor_names:
            self.config.enable_stream(rs.stream.infrared, 2,
                                      self.width["IR2"], self.height["IR2"],
                                      rs.format.y8, self.fps["IR2"])
        if "DEPTH" in self.sensor_names:
            self.config.enable_stream(rs.stream.depth, 0,
                                      self.width["DEPTH"], self.height["DEPTH"],
                                      rs.format.z16, self.fps["DEPTH"])

        # If the Depth stream is enabled, configure the depth sensor. Otherwise, disable the emitter.
        depth_sensor = self.dev.first_depth_sensor()
        rgb_sensor = None
        for s in self.dev.query_sensors():
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                rgb_sensor = s
                break

        want_emitter = True if "DEPTH" in self.sensor_names else bool(enable_projection)
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if want_emitter else 0.0)

        # Setting the exposure for each sensor if specified
        # --- IR (Depth sensor) auto/manual exposure once ---
        ir_exp_1 = self.exposure.get("IR1")
        ir_exp_2 = self.exposure.get("IR2")
        ir_manual = (ir_exp_1 is not None) or (ir_exp_2 is not None)
        if depth_sensor and depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0.0 if ir_manual else 1.0)
        # depth/IR share the same sensor -> ONE exposure value
        if ir_manual and depth_sensor and depth_sensor.supports(rs.option.exposure):
            ir_exp = ir_exp_1 if ir_exp_1 is not None else ir_exp_2
            depth_sensor.set_option(rs.option.exposure, float(ir_exp))

        if "RGB" in self.sensor_names:
            rgb_exp = self.exposure.get("RGB")
            if rgb_sensor and rgb_sensor.supports(rs.option.enable_auto_exposure):
                rgb_sensor.set_option(rs.option.enable_auto_exposure, 0.0 if rgb_exp is not None else 1.0)
            if rgb_exp is not None and rgb_sensor and rgb_sensor.supports(rs.option.exposure):
                rgb_sensor.set_option(rs.option.exposure, float(rgb_exp))
    
    def change_exposure(self, sensor_name: str, exposure_value: float = None):
        '''
        cam.change_exposure("IR1", 8000)   # IR1/IR2 manual 8000us
        cam.change_exposure("RGB", None)   # RGB restore auto exposure
        '''
        s = sensor_name.upper()
        if s not in {"RGB", "IR1", "IR2"}:
            raise ValueError("sensor_name must be RGB / IR1 / IR2")

        def _clamp(val, lo, hi):
            return max(lo, min(hi, val))

        if s in {"IR1", "IR2"}:
            depth_sensor = self.dev.first_depth_sensor()
            if not depth_sensor:
                raise RuntimeError("No depth/IR sensor found.")

            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1.0 if exposure_value is None else 0.0)
            if exposure_value is not None and depth_sensor.supports(rs.option.exposure):
                # RealSense normal exposure range: ~1-20000+ µs, varies slightly by firmware
                exp = float(_clamp(exposure_value, 1.0, 200000.0))
                depth_sensor.set_option(rs.option.exposure, exp)
                return exp
            return None

        # --- RGB path ---
        for cam in self.dev.query_sensors():
            if cam.get_info(rs.camera_info.name) == "RGB Camera":
                if cam.supports(rs.option.enable_auto_exposure):
                    cam.set_option(rs.option.enable_auto_exposure, 1.0 if exposure_value is None else 0.0)
                if exposure_value is not None and cam.supports(rs.option.enable_auto_white_balance):
                    cam.set_option(rs.option.enable_auto_white_balance, 0.0)
                if exposure_value is None and cam.supports(rs.option.enable_auto_white_balance):
                    cam.set_option(rs.option.enable_auto_white_balance, 1.0)
                if cam.supports(rs.option.power_line_frequency):
                    try:
                        cam.set_option(rs.option.power_line_frequency, 3.0)  # 3=50Hz
                    except Exception:
                        pass

                if exposure_value is not None and cam.supports(rs.option.exposure):
                    exp = float(_clamp(exposure_value, 1.0, 200000.0))
                    cam.set_option(rs.option.exposure, exp)
                    return exp
                return None
        raise RuntimeError("RGB sensor not found.")
    

    def start_pipeline(self) -> rs.pipeline_profile:
        """
        Start the RealSense pipeline with the configured settings.
        """
        profile = self.pipeline.start(self.config)
        # print("Pipeline started with configuration:", self.config)
        return profile
    
    def stop_pipeline(self):
        """
        Stop the RealSense pipeline.
        """
        self.pipeline.stop()

    def get_frames(self):
        """
        Get frames from the RealSense pipeline.
        """
        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)
        for name in self.sensor_names:
            if name == "RGB":
                self.frame_cache["RGB"] = frames.get_color_frame()
            elif name == "IR1":
                self.frame_cache["IR1"] = frames.get_infrared_frame(1)
            elif name == "IR2":
                self.frame_cache["IR2"] = frames.get_infrared_frame(2)
            elif name == "DEPTH":
                self.frame_cache["DEPTH"] = frames.get_depth_frame()
        return self.frame_cache

    def convert_frame_to_array(self, frame) -> np.ndarray | None:
        """
        Convert a RealSense frame to a NumPy array.
        """
        if frame:
            return np.asanyarray(frame.get_data())
        return None

    def convert_frame_to_image(self, sensor_name: str, frame):
        """
        Convert a RealSense frame to a format suitable for saving as an image.
        """
        assert sensor_name in self.sensor_names, f"Sensor {sensor_name} not initialized."
        if sensor_name == "RGB":
            return np.asanyarray(frame.get_data())
        if sensor_name == "IR1":
            return cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_GRAY2BGR)
        if sensor_name == "IR2":
            return cv2.cvtColor(np.asanyarray(frame.get_data()), cv2.COLOR_GRAY2BGR)
        if sensor_name == "DEPTH":
            colorizer = rs.colorizer()
            return np.asanyarray(colorizer.colorize(frame).get_data())
        return None

class CalibImageSaver:
    def __init__(self,
                 sensor_name: str = "RGB",
                 width: int = 1280, height: int = 720, fps: int = 30,
                 exposure_us: int | None = None,
                 base_dir: Path | None = None,
                 checkerboard: tuple[int, int] = (11, 8),  
                 square_size_mm: int = 15,                    
                 save_depth_npy: bool = False,
                 fname_template: str | None = None):
        s = sensor_name.upper()
        assert s in ("RGB", "IR1", "IR2"), "sensor_name should be 'RGB', 'IR1', or 'IR2'"

        self.sensor_name = s
        self.cam = CameraInit(
            sensor_names=s,
            width=width,
            height=height,
            fps=fps,
            exposure=exposure_us
        )
        self.pipeline = self.cam.pipeline
        self.config = self.cam.config

        base_dir = base_dir if base_dir is not None else Path(__file__).resolve().parent
        if "IR" in s:
            self.save_dir = base_dir / f"{s.lower()}_calib_images"
        else:
            self.save_dir = base_dir / "rgb_calib_images"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save directory: {self.save_dir}")

        if fname_template is None:
            self.fname_template = f"{self.sensor_name.lower()}_" + "{:03d}.jpg"
        else:
            self.fname_template = fname_template

        self.win_name   = f"{s} Preview"
        self.img_count  = 0
        self.t_prev     = None


        self.CHECKERBOARD   = checkerboard
        self.SQUARE_SIZE_MM = square_size_mm
        self.SAVE_DEPTH_NPY = save_depth_npy

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        self.cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                         cv2.CALIB_CB_FAST_CHECK |
                         cv2.CALIB_CB_NORMALIZE_IMAGE)

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def start(self):
        self.cam.start_pipeline()
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        print("Press <Space> to save (only if chessboard detected), press <ESC>/<q> to quit.")
        self.t_prev = time.perf_counter()

    def stop(self):
        self.cam.stop_pipeline()
        cv2.destroyAllWindows()

    def run(self):
        try:
            while True:
                frames = self.cam.get_frames()
                frame  = frames.get(self.sensor_name, None)
                if frame is None:
                    continue

                img = self.cam.convert_frame_to_image(self.sensor_name, frame)
                if img.ndim == 2:
                    gray = img
                    disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    disp = img.copy()

                # --- 棋盘格检测 ---
                found, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, self.cb_flags)
                if found:
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(disp, self.CHECKERBOARD, corners, found)
                    status_txt = "Chessboard detected  (SPACE=save)"
                    status_col = (0, 255, 0)
                else:
                    status_txt = "No chessboard  (adjust pose)"
                    status_col = (0, 0, 255)

                # --- FPS & 状态覆盖 ---
                now = time.perf_counter()
                fps = 1.0 / (now - self.t_prev) if self.t_prev is not None else 0.0
                self.t_prev = now
                h, w = disp.shape[:2]
                cv2.putText(disp, f"FPS: {fps:.1f}", (15, 40), self.font, 1, (0, 255, 0), 2)
                cv2.putText(disp, status_txt, (15, h - 20), self.font, 0.8, status_col, 2)

                cv2.imshow(self.win_name, disp)
                key = cv2.waitKey(1) & 0xFF

                if key == 32:  # Space
                    if found:
                        fname = self.fname_template.format(self.img_count)
                        path  = self.save_dir / fname
                        cv2.imwrite(str(path), img)
                        print(f"[✓] Saved {path.name}")

                        if self.SAVE_DEPTH_NPY:
                            depth_frame = frames.get("DEPTH", None)
                            if depth_frame is not None:
                                depth = self.cam.convert_frame_to_image("DEPTH", depth_frame)
                                np.save(self.save_dir / f"{Path(fname).stem}_depth.npy", depth)
                        self.img_count += 1
                    else:
                        print("[×] Can not find chessboard corners.")

                if key in (27, ord('q')):  # ESC 或 q
                    print("Exiting capture.")
                    return
        finally:
            self.stop()

class CameraIntrinsicCalibrator:
    def __init__(
        self,
        img_pattern: str | Path,
        checkerboard: Tuple[int, int] = (11, 8),
        square_size: float = 15,
        visualize: bool = False,
        yaml_path: str | Path = "camera_intrinsic.yml"
    ) -> None:
        self.checkerboard = checkerboard
        self.square_size = float(square_size)
        self.visualize = visualize
        self.yaml_path = Path(yaml_path)

        # change img_pattern to a glob pattern if it's a directory
        # assert whether pattern path exists
        assert img_pattern.exists(), "Image pattern path does not exist."
        if isinstance(img_pattern, Path):
            if img_pattern.is_dir():
                self.img_pattern = str(img_pattern / "*.jpg")
            else:
                self.img_pattern = str(img_pattern)
        else:
            self.img_pattern = str(img_pattern)


        # corner detection parameters
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 1e-3
        )
        self._cb_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH |
            cv2.CALIB_CB_FAST_CHECK |
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # data caching
        self._objpoints: List[np.ndarray] = []
        self._imgpoints: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None  # (w, h)

        # results
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.rvecs: Optional[List[np.ndarray]] = None
        self.tvecs: Optional[List[np.ndarray]] = None
        self.rms: Optional[float] = None
        self.mean_reproj_err: Optional[float] = None

        # pre-generate world coordinate template
        cols, rows = self.checkerboard
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        self._objp_template = objp * self.square_size

        # automatic calibration
        n = self._load_images()
        print(f"Effective images: {n}")
        if n > 0:
            self._calibrate()
            self._save_yaml()
            print(f"Camera intrinsics saved to {self.yaml_path}")
        else:
            print("No valid images found for calibration.")

    def _load_images(self) -> int:
        count = 0
        for fname in glob.glob(self.img_pattern):
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if self._add_image(img):
                count += 1
        return count
    
    def _add_image(self, img_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, self.checkerboard, self._cb_flags
        )
        if not ret:
            if self.visualize:
                self._show(img_bgr, text="No chessboard")
            return False

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)
        self._imgpoints.append(corners)
        self._objpoints.append(self._objp_template.copy())
        self._image_size = (gray.shape[1], gray.shape[0])

        if self.visualize:
            vis = img_bgr.copy()
            cv2.drawChessboardCorners(vis, self.checkerboard, corners, True)
            self._show(vis, text="Detected")
        return True
    
    def _calibrate(self) -> None:
        if not self._objpoints:
            raise RuntimeError("No valid chessboard observations found for calibration.")

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._objpoints, self._imgpoints, self._image_size, None, None
        )

        self.K, self.dist = K, dist
        self.rvecs, self.tvecs = rvecs, tvecs
        self.rms = float(rms)
        self.mean_reproj_err = self._compute_mean_reproj_error()

        print("RMSE:", self.rms)
        # print("K:\n", self.K)
        # print("dist:", self.dist.ravel())
        # print("Mean reprojection error:", self.mean_reproj_err)

    def _compute_mean_reproj_error(self) -> float:
        total_err = 0.0
        n = len(self._objpoints)
        for i in range(n):
            imgpoints2, _ = cv2.projectPoints(
                self._objpoints[i], self.rvecs[i], self.tvecs[i], self.K, self.dist
            )
            err = cv2.norm(self._imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_err += err
        return total_err / n
    
    def _save_yaml(self) -> None:
        fs = cv2.FileStorage(str(self.yaml_path), cv2.FILE_STORAGE_WRITE)
        fs.write("K", self.K)
        fs.write("dist", self.dist)
        fs.release()

    @staticmethod
    def load_from_yaml(yaml_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
        fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
        K = fs.getNode("K").mat()
        dist = fs.getNode("dist").mat()
        fs.release()
        return K, dist
    
    def _show(self, img_bgr: np.ndarray, text: str = "") -> None:
        disp = img_bgr.copy()
        if text:
            cv2.putText(disp, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Calib Preview", disp)
        cv2.waitKey(30)

    @staticmethod
    def get_device_intrinsics(width: int = 1280, height: int = 720, fps: int = 30):
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        align = rs.align(rs.stream.color)
        pipeline.start(cfg)
        try:
            frames = align.process(pipeline.wait_for_frames())
            c = frames.get_color_frame()
            if c: 
                intr = c.profile.as_video_stream_profile().intrinsics
                K = np.array([[intr.fx, 0, intr.ppx],
                            [0, intr.fy, intr.ppy],
                            [0,       0,       1]], dtype=np.float64)
                dist = np.array(intr.coeffs, dtype=np.float64).reshape(-1,1)
                # print(f"RealSense intrinsics: {K}, dist: {dist}")
                return K, dist, intr
            raise RuntimeError("Failed to get RealSense intrinsics")
        finally:
            pipeline.stop()

class CameraExternalCalibrator:
    def __init__(
        self,
        K: np.ndarray, dist: np.ndarray,
        checkerboard: Tuple[int, int] = (11, 8),
        square_size: float = 15.0,
        yaml_path: str | Path = "t_T_cam.yml",
        visualize: bool = True,
    ) -> None:
        """
        K, dist: 相机内参与畸变（来自你的 CameraIntrinsicCalibrator 或 RealSense intrinsics）
        checkerboard: (cols, rows) = 内角点数
        square_size: 每格边长（mm）
        yaml_path: 输出路径
        """
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64)
        self.checkerboard = checkerboard
        self.square_size = float(square_size)
        self.yaml_path = Path(yaml_path)
        self.visualize = visualize

        # 生成棋盘 3D 物点（Z=0，原点在左上角内角点）
        cols, rows = checkerboard
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        self.objp = objp * self.square_size

        # 缓存求平均
        self._rvecs: List[np.ndarray] = []
        self._tvecs: List[np.ndarray] = []

    @staticmethod
    def _rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.reshape(3)
        return T

    @staticmethod
    def _invert_T(T: np.ndarray) -> np.ndarray:
        R = T[:3, :3]; t = T[:3, 3:4]
        Ti = np.eye(4, dtype=np.float64)
        Ti[:3, :3] = R.T
        Ti[:3, 3:4] = -R.T @ t
        return Ti

    def _solve_pose_from_image(self, img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_FAST_CHECK |
                 cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, flags)
        if not ret:
            return None

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))

        # 物体→相机：camTt = [R|t]（solvePnP 的 rvec,tvec 是 "t->cam"）
        ok, rvec, tvec = cv2.solvePnP(self.objp, corners, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        camTt = self._rt_to_T(R, tvec)

        tTcam = self._invert_T(camTt)

        if self.visualize:
            vis = img_bgr.copy()
            cv2.drawChessboardCorners(vis, self.checkerboard, corners, True)
            axis_len = self.square_size * 3
            axis_obj = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,-axis_len]])  # 注意右手：Z朝外屏幕
            imgpts, _ = cv2.projectPoints(axis_obj, rvec, tvec, self.K, self.dist)
            imgpts = imgpts.reshape(-1,2).astype(int)
            cv2.line(vis, tuple(imgpts[0]), tuple(imgpts[1]), (0,0,255), 3)   # X 红
            cv2.line(vis, tuple(imgpts[0]), tuple(imgpts[2]), (0,255,0), 3)   # Y 绿
            cv2.line(vis, tuple(imgpts[0]), tuple(imgpts[3]), (255,0,0), 3)   # Z 蓝（指向相机外）
            cv2.putText(vis, "PnP OK (SPACE=accept)", (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("ExtCalib", vis); cv2.waitKey(1)

        return tTcam, rvec, tvec

    def calibrate_from_images(self, img_glob: str | Path) -> np.ndarray:
        paths = sorted(glob.glob(str(img_glob)))
        assert paths, f"No images matched: {img_glob}"

        self._rvecs.clear(); self._tvecs.clear()
        Ts = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None: 
                continue
            out = self._solve_pose_from_image(img)
            if out is None:
                continue
            tTcam, rvec, tvec = out
            Ts.append(tTcam)
            self._rvecs.append(rvec.reshape(3))
            self._tvecs.append(tvec.reshape(3))

        if not Ts:
            raise RuntimeError("No valid chessboard pose found.")

        tTcam_avg = self._average_poses(self._rvecs, self._tvecs)
        self._save_yaml(tTcam_avg, len(Ts))
        return tTcam_avg

    def calibrate_from_camera(self, cam) -> np.ndarray:
        """
        cam: 你的 CameraInit 实例（必须包含已启动的 pipeline）
        操作：看到棋盘后按空格采一张；可多采几张做平均；按 q 结束并保存。
        """
        print("Show chessboard to camera. <Space>=add sample, <q>=finish")
        self._rvecs.clear(); self._tvecs.clear()
        samples = 0
        while True:
            frames = cam.get_frames()
            frame = frames.get("RGB") or frames.get("IR1") or frames.get("IR2")
            if frame is None: 
                continue
            img = cam.convert_frame_to_image(
                "RGB" if "RGB" in cam.sensor_names else ("IR1" if "IR1" in cam.sensor_names else "IR2"),
                frame
            )
            out = self._solve_pose_from_image(img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if out is not None:
                    tTcam, rvec, tvec = out
                    self._rvecs.append(rvec.reshape(3))
                    self._tvecs.append(tvec.reshape(3))
                    samples += 1
                    print(f"[+] sample {samples}")
                else:
                    print("[x] chessboard not found")
            if key in (ord('q'), 27):
                break

        if samples == 0:
            raise RuntimeError("No samples collected.")
        tTcam_avg = self._average_poses(self._rvecs, self._tvecs)
        self._save_yaml(tTcam_avg, samples)
        return tTcam_avg

    # ---------- 姿态平均（罗德里格斯→四元数 + 平均平移） ----------
    def _average_poses(self, rvecs: List[np.ndarray], tvecs: List[np.ndarray]) -> np.ndarray:
        def rvec_to_quat(rvec):
            R,_ = cv2.Rodrigues(rvec.reshape(3,1))
            # w,x,y,z
            qw = np.sqrt(1.0 + np.trace(R)) / 2.0
            qx = (R[2,1]-R[1,2])/(4*qw); qy = (R[0,2]-R[2,0])/(4*qw); qz = (R[1,0]-R[0,1])/(4*qw)
            return np.array([qw,qx,qy,qz])
        def quat_to_R(q):
            qw,qx,qy,qz = q
            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1-2*(qx*qx+qy*qy)]
            ], dtype=np.float64)
            return R

        Q = np.stack([rvec_to_quat(r) for r in rvecs], 0)
        # 方向一致化
        for i in range(1,len(Q)):
            if np.dot(Q[0], Q[i]) < 0: Q[i] = -Q[i]
        q_mean = Q.mean(0); q_mean = q_mean / np.linalg.norm(q_mean)
        R_mean = quat_to_R(q_mean)
        t_mean = np.mean(np.stack(tvecs,0),0)

        camTt_mean = self._rt_to_T(R_mean, t_mean)
        tTcam_mean = self._invert_T(camTt_mean)
        return tTcam_mean

    def _save_yaml(self, tTcam: np.ndarray, nviews: int) -> None:
        R = tTcam[:3,:3]
        t = tTcam[:3,3].reshape(3,1)
        fs = cv2.FileStorage(str(self.yaml_path), cv2.FILE_STORAGE_WRITE)
        fs.write("t_T_cam", tTcam)
        fs.write("R_cam_to_t", R)
        fs.write("t_cam_in_t", t)
        fs.write("checkerboard_cols_rows", np.array(self.checkerboard, dtype=np.int32))
        fs.write("square_size_mm", float(self.square_size))
        fs.write("n_views", int(nviews))
        fs.write("K", self.K); fs.write("dist", self.dist)
        fs.release()
        print(f"[✓] Saved t_T_cam to {self.yaml_path}")


    @staticmethod
    def load_from_yaml(yaml_path: str | Path) -> np.ndarray:
        """
        从保存的 t_T_cam.yaml 读取外参矩阵
        返回: t_T_cam (4x4 numpy array, dtype=float64)
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
        t_T_cam = fs.getNode("t_T_cam").mat()
        fs.release()

        if t_T_cam is None or t_T_cam.shape != (4,4):
            raise ValueError(f"Invalid t_T_cam in YAML: {yaml_path}")

        return t_T_cam

if __name__ == "__main__":
    pass
    # cam = CameraInit(
    #     sensor_names=["RGB", "IR1", "IR2"],
    #     width={"RGB": 1280, "IR1": 1280, "IR2": 1280, "DEPTH": 1280},
    #     height={"RGB": 720, "IR1": 720, "IR2": 720, "DEPTH": 720},
    #     fps={"RGB": 30, "IR1": 30, "IR2": 30, "DEPTH": 30},
    #     exposure={"RGB": 200, "IR1": 1000, "IR2": 1000}
    # )

    # # Start the pipeline
    # cam.start_pipeline()
    # # print camera properties
    # print("Camera Widths:", cam.width)
    # print("Camera Heights:", cam.height)
    # print("Camera FPS:", cam.fps)
    # print("Camera Exposure:", cam.exposure)

    # def no_signal_frame(w, h, text="NO SIGNAL"):
    #     img = np.full((h, w, 3), 80, dtype=np.uint8)
    #     cv2.putText(img, text, (w//6, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)
    #     return img
    
    # try:
    #     while True:

    #         frames = cam.get_frames()

    #         top = np.hstack((cam.convert_frame_to_image("RGB", frames["RGB"]),
    #                          cam.convert_frame_to_image("IR1", frames["IR1"])))
    #         bottom = np.hstack((cam.convert_frame_to_image("IR2", frames["IR2"]),
    #                            no_signal_frame(1280, 720)))
    #         grid = np.vstack((top, bottom))

    #         cv2.imshow("Multi-Stream Test", grid)
    #         if cv2.waitKey(1) & 0xFF == 27:
    #             break
    #         # change the exposure
    #         elif cv2.waitKey(1) & 0xFF == ord('e'):
    #             print("Changing exposure settings...")
    #             cam.change_exposure("RGB", 2000)  # Set RGB exposure to 2000us
    #             cam.change_exposure("IR1", 10)
    #             cam.change_exposure("IR2", 10)

    # finally:
    #     cam.stop_pipeline()
    #     cv2.destroyAllWindows()

    ################################### Save
    # RGB 
    # saver = CalibImageSaver(sensor_name="RGB")
    # saver.start()
    # saver.run()

    ################################## Camera Calibration

    # calib = CameraIntrinsicCalibrator(
    #     img_pattern=Path("./rgb_calib_images"),
    #     checkerboard=(11, 8),
    #     square_size=15,
    #     visualize=True,
    #     yaml_path="camera_intrinsic.yml"
    # )

    # K, dist = CameraIntrinsicCalibrator.load_from_yaml("camera_intrinsic.yml")
    # print("Loaded K:\n", K)
    # print("Loaded dist:", dist.ravel())

    # # Recommend to use the CameraIntrinsicCalibrator.get_device_intrinsics() method from the device directly, because intel has already calibrated the camera intrinsics.
    # CameraIntrinsicCalibrator.get_device_intrinsics()

    ################################### Extrinsic Calibration

    # 从设备直接获取内参（K, dist）
    K, dist, _ = CameraIntrinsicCalibrator.get_device_intrinsics(width=1280, height=720, fps=30)

    # 初始化外参标定器
    ext = CameraExternalCalibrator(
        K, dist,
        checkerboard=(11, 8),   # 你的棋盘内角点数
        square_size=15,         # 每格 mm
        yaml_path="t_T_cam.yml"
    )

    # 实时采集标定外参
    cam = CameraInit(sensor_names="RGB", width=1280, height=720, fps=30, exposure=None)
    cam.start_pipeline()
    try:
        t_T_cam = ext.calibrate_from_camera(cam)  # 空格采样，q 保存退出
    finally:
        cam.stop_pipeline()
        cv2.destroyAllWindows()


    # read the calibration result
    t_T_cam = CameraExternalCalibrator.load_from_yaml("t_T_cam.yml")
    print("Loaded t_T_cam:\n", t_T_cam)
