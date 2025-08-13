import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import json
from CameraCalibration import *


reference_points = {
    'A': (0.00,  55.00,  6.0),
    'B': (0.00,  30.00,  6.0),
    'C': (0.00, -25.00,  6.0),
    'D': (17.68, 17.68, 12.0),
    'E': (-24.75,-24.75, 12.0),
}

class MarkerDetector:

    def __init__(self, reference_points: dict[str, tuple], width: int = 1280, height: int = 720, exposure: int = 500, enable_double: bool = False):
        """
        Initialize the MarkerDetector with reference points.
        """
        self.reference_points = reference_points
        self.enable_double = enable_double
        # Init the camera steam
        if self.enable_double:
            self.cam = CameraInit(
                sensor_names=["IR1", "IR2"],
                width={"IR1": width, "IR2": width},
                height={"IR1": height, "IR2": height},
                exposure={"IR1": exposure, "IR2": exposure}
            )
        else:
            self.cam = CameraInit(
                sensor_names=["IR1"],
                width={"IR1": width},
                height={"IR1": height},
                exposure={"IR1": exposure}
            )
        # Start the camera
        self.cam.start_pipeline()
        self._thread = None

    def __del__(self):
        self.cam.stop_pipeline()

    def smooth_signal(self, image: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 1.0) -> np.ndarray:
        """
        Apply a low-pass filter to the image to reduce noise and smooth the image.

        Parameters:
        image (np.ndarray): Input image.

        Returns:
        smoothed_image (np.ndarray): Image with low-frequency content reduced.
        """
        # Define a Gaussian kernel for low-pass filtering
        kernel = cv2.GaussianBlur(image, kernel_size, sigma)
        return kernel
    
    def extract_ir_high_freq(self, ir_image):
        """
        Apply a high-pass (Laplacian) filter to the IR image to extract high-frequency components.

        Parameters:
        ir_image (np.ndarray): Single-channel 8-bit IR image.

        Kernel definition (3×3 Laplacian operator):
            [ -1, -1, -1 ]
            [ -1,  8, -1 ]
            [ -1, -1, -1 ]
        - Center weight of 8 balances the sum of neighbor weights to zero, preserving average brightness.
        - Negative neighbor weights suppress low-frequency areas, emphasizing edges and fine details.

        filter2D arguments:
        - src     : input image (ir_image)
        - ddepth  : -1 (same depth as source, produces 8-bit output)
        - kernel  : convolution kernel defined above
        - anchor  : default (-1, -1) places the anchor at the kernel center

        Returns:
        high_freq (np.ndarray): Image with high-frequency content emphasized.
        """
        # Define 3x3 Laplacian kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        # Apply high-pass filter; ddepth=-1 keeps same data type
        high_freq = cv2.filter2D(src=ir_image, ddepth=-1, kernel=kernel)
        return high_freq
    
    def threshold_image(self, image, threshold_min=200, threshold_max=255):
        """
        Apply a binary threshold to the image to isolate high-frequency components.

        Parameters:
        image (np.ndarray): Input image.
        threshold_min (int): Minimum threshold value.
        threshold_max (int): Maximum threshold value.

        Returns:
        binary_image (np.ndarray): Binary image after applying the threshold.
        """
        # Apply binary thresholding
        _, binary_image = cv2.threshold(image, threshold_min, threshold_max, cv2.THRESH_BINARY)
        return binary_image

    def process_frame(self, frame):
        # Implement frame processing logic here
        # Get the raw frame data
        if self.enable_double:
            raw_frame_ir1 = self.cam.get_frames()["IR1"]
            raw_frame_ir2 = self.cam.get_frames()["IR2"]
        else:
            raw_frame_ir1 = self.cam.get_frames()["IR1"]
            raw_frame_ir2 = None

        # 正确赋值 raw_array_ir1 和 raw_array_ir2
        if raw_frame_ir1 is not None:
            raw_array_ir1 = self.cam.convert_frame_to_array(raw_frame_ir1)
        else:
            raw_array_ir1 = None
        if raw_frame_ir2 is not None:
            raw_array_ir2 = self.cam.convert_frame_to_array(raw_frame_ir2)
        else:
            raw_array_ir2 = None

        # 检查 raw_array_ir1 是否有效
        if raw_array_ir1 is None:
            return None

        # Process the raw frame data
        result = self.smooth_signal(raw_array_ir1)
        result = self.extract_ir_high_freq(result)


        return result
    
    def start_threaded(self):
        """
        Start the marker detection in a separate thread.
        """
        import threading
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self._thread
    
    def run(self):
        """
        Main loop to capture and process frames from the camera.
        """
        try:
            while True:
                # Capture a frame
                frame = self.cam.get_frames()
                if not frame:
                    continue
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow('Processed Frame', processed_frame)

                # Exit on 'q' key press or ESC key
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
                    break

        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            cv2.destroyAllWindows()


# main
if __name__ == "__main__":
    marker_detector = MarkerDetector(reference_points)
    marker_detector.start_threaded()
    try:
        while True:
            if not marker_detector._thread.is_alive():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Main thread interrupted.")
