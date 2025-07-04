import pyrealsense2 as rs
import numpy as np
import cv2
import time
from kalman_filter import MarkerTracker
from outlier_detection import RobustMarkerDetector

class MarkerInit:

    def __init__(self, debug=False):

        self.DEBUG = debug

        ############
        # in order to make the result perfect we need some strategy
        # 1. The circle confidence would be change order, we need to make sure A, B, C, D identify could be tracked correctly - kalman_filter.py
        # 2. We could include the kalman filter to make the result more stable we need to make sure each frame has relationship in time series. - kalman_filter.py
        # 3. Also for the abnormal value detection. - outlier_detection.py
        # 4. We need to has some module to auto calibrate the image, like exposure, binary threshold, etc.

        # Marker real space location in millimeters
        # optical marker version 3.0
        # 
        #        D
        #         -
        #          -
        #           -
        #     B------*---------A
        #             -
        #              -
        #               C
        # 
        self.reference_points = {
            'A': (0, 40, 7),
            'B': (0, -30, 7),
            'C': (30.31, 17.5, 13),
            'D': (-38.97, -22.5, 13)
        }

        # Create pipeline, config, and context
        self.pipeline = rs.pipeline()

        self.config   = rs.config()
        self.ctx      = rs.context()

        # Query connected devices
        self.devices = self.ctx.query_devices()
        if not self.devices:
            raise RuntimeError("No RealSense devices found.")
        
        # Select the first device
        self.dev = self.devices[0]
        # Print device information
        print(f"Found device:          {self.dev.get_info(rs.camera_info.name)}")
        print(f"Serial Number:         {self.dev.get_info(rs.camera_info.serial_number)}")
        print(f"Firmware Version:      {self.dev.get_info(rs.camera_info.firmware_version)}\n")

        # Print system information
        # self.list_all_adjustable_params()

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
        # │ Infrared   │ 0     │ 640×480    │ y8       │ 6, 15, 30        │
        # │ Infrared   │ 1     │ 640×480    │ y8       │ 6, 15, 30        │
        # │ Infrared   │ 0     │ 1280×720   │ y8       │ 6, 15, 30        │
        # │ Infrared   │ 1     │ 1280×720   │ y8       │ 6, 15, 30        │
        # └────────────┴───────┴────────────┴──────────┴──────────────────┘
        #
        # Usage examples:
        #    config.enable_stream(rs.stream.depth,      0, 640, 480, rs.format.z16, 30)
        #    config.enable_stream(rs.stream.color,      0, 1280, 720, rs.format.rgb8, 15)
        #    config.enable_stream(rs.stream.infrared,   1, 640, 480, rs.format.y8,   30)
        #
        self.config.enable_stream(
            rs.stream.infrared, # Stream type: Infrared
            1,                  # Stream index: second IR sensor (0=left, 1=right)
            1280,               # Width: 1280 px
            720,                # Height: 720 px
            rs.format.y8,       # Format: Y8 (8-bit grayscale)
            30                  # Frame rate: 30 FPS
        )

        # close the IR projection
        depth_sensor = self.dev.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        # if depth_sensor.supports(rs.option.laser_power):
        #     depth_sensor.set_option(rs.option.laser_power, 0)

        # Set the exposure for the infrared sensor
        if depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, 2000)

        profile = self.pipeline.start(self.config)

        # Print configured infrared stream details
        print("Configured Infrared Stream:")
        for stream_profile in profile.get_streams():
            if stream_profile.stream_type() == rs.stream.infrared:
                vsp = stream_profile.as_video_stream_profile()
                intr = vsp.get_intrinsics()
                print(f"  Stream: {vsp.stream_name()} @ {vsp.fps()} FPS")
                print(f"    Resolution: {intr.width} x {intr.height}")
                print(f"    Intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}, ppx={intr.ppx:.2f}, ppy={intr.ppy:.2f}")
                print(f"    Distortion: {intr.model.name}, Coeffs={intr.coeffs}\n")

        # Stop the pipeline
        self.pipeline.stop()
        print("Initialization complete.")

        # Initialize the marker tracker and outlier detector
        self.marker_tracker = MarkerTracker(['A', 'B', 'C', 'D'])
        self.outlier_detector = RobustMarkerDetector(['A', 'B', 'C', 'D'])

        # Initialize tracking state
        self.frame_count = 0
        self.tracking_started = False
        self.successful_tracking_count = 0


    def __del__(self):
        # Ensure pipeline is stopped on deletion
        try:
            self.pipeline.stop()
        except RuntimeError:
            print("Pipeline already stopped or not started.")

    def list_all_adjustable_params(self):
        '''
        List all adjustable parameters for each device.
        '''
        # List all adjustable options per sensor
        for sensor in self.dev.query_sensors():
            name = sensor.get_info(rs.camera_info.name)
            opts = sensor.get_supported_options()
            if not opts:
                print(f"{name}: No adjustable options.\n")
                continue
            print(f"{name}: Adjustable options:")
            for opt in opts:
                try:
                    r   = sensor.get_option_range(opt)
                    val = sensor.get_option(opt)
                except RuntimeError:
                    continue
                print(f"  {opt.name}: {val:.3f}, Range=[{r.min:.3f}, {r.max:.3f}], Step={r.step:.3f}")
            print()

    def morphological_operations(self, image, kernel_open_size=(3, 3), kernel_close_size=(5, 5), iterations=1):
        """
        Apply morphological operations (dilation and erosion) to the image to enhance features.

        Parameters:
        image (np.ndarray): Input binary image.
        kernel_size (tuple): Size of the structuring element for morphological operations.
        iterations (int): Number of iterations for dilation and erosion.

        Returns:
        morphed_image (np.ndarray): Image after applying morphological operations.
        """
        # Create a structuring element
        morph = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones(kernel_open_size, np.uint8), iterations=iterations)
        # Apply closing operation to fill small holes
        morphed_image = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, np.ones(kernel_close_size, np.uint8), iterations=iterations)

        return morphed_image

    def smooth_signal(self, image, kernel_size=(7, 7), sigma=1):
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

    def fit_circle_ls(self, points):
        """
        Fit a circle to a set of points using least squares method.
        Minimizes the algebraic distance to the circle.

        Parameters:
        points (np.ndarray): N×2 array of contour points

        Returns:
        (xc, yc, r): Fitted circle's center and radius
        """
        x = points[:, 0]
        y = points[:, 1]
        x_m, y_m = x.mean(), y.mean()
        u = x - x_m
        v = y - y_m
        Suu = np.sum(u * u)
        Suv = np.sum(u * v)
        Svv = np.sum(v * v)
        Suuu = np.sum(u * u * u)
        Svvv = np.sum(v * v * v)
        Suvv = np.sum(u * v * v)
        Svuu = np.sum(v * u * u)

        # Construct linear equations A * [uc, vc] = B
        A = np.array([[Suu, Suv],
                      [Suv, Svv]])
        B = np.array([0.5 * (Suuu + Suvv),
                      0.5 * (Svvv + Svuu)])

        # Solve for eccentricity coordinates (uc, vc)
        try:
            uc, vc = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return None  # Singular matrix cannot fit

        xc = x_m + uc
        yc = y_m + vc
        # Use mean distance as radius
        r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2).mean()
        return xc, yc, r

    def simple_circle_detection(self, image, min_area=10, max_area=8000, circularity_threshold=0.5):
        """
        Detect circles in a binary image. For incomplete circular arcs, fit a complete circle using least squares method.

        Parameters:
        image (np.ndarray): Binary input image
        min_area (int): Minimum contour area
        max_area (int): Maximum contour area
        circularity_threshold (float): Circularity threshold (0-1)

        Returns:
        circles (list): Detected circles [(cx, cy, r, confidence), ...]
        """
        circles = []
        # Find external contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # Check if it is an approximately complete circle
            if circularity >= circularity_threshold:
                # Use minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
            else:
                # Try fitting an incomplete circular arc
                pts = contour.reshape(-1, 2)
                fit = self.fit_circle_ls(pts)
                if fit is None:
                    continue
                x, y, radius = fit

                # Optional: Further filtering based on fitting error and area range
                # mean_error = np.abs(np.sqrt((pts[:,0]-x)**2 + (pts[:,1]-y)**2) - radius).mean()
                # if mean_error > some_threshold: continue

            # Calculate detection confidence (user-defined method)
            confidence = self.calculate_circle_confidence(contour, area, perimeter, circularity, radius)
            circles.append((int(x), int(y), int(radius), confidence))

        if self.DEBUG:
            print(f"Detected {len(circles)} circles.")

        # Sort by confidence and return top 4
        circles.sort(key=lambda x: x[3], reverse=True)
        return circles[:4]    

    def calculate_circle_confidence(self, contour, area, perimeter, circularity, radius):
        """
        Calculate the confidence of circle detection based on contour properties.
        
        Parameters:
        contour (np.ndarray): contour of the detected shape
        area (float): area of the contour
        perimeter (float): perimeter of the contour
        circularity (float): circularity of the contour
        radius (float): radius of the minimum enclosing circle

        Returns:
        confidence (float): confidence score for the circle detection
        """
        confidence = 0.0

        # 1. circularity weight (40%)
        circularity_score = min(circularity, 1.0)  # make sure circularity is between 0 and 1
        confidence += circularity_score * 0.4
        
        # 2. area weight (20%)
        # ideal area range for a circle
        ideal_area_min, ideal_area_max = 10, 8000
        if ideal_area_min <= area <= ideal_area_max:
            area_score = 1.0
        elif area < ideal_area_min:
            area_score = area / ideal_area_min
        else:
            area_score = ideal_area_max / area
        confidence += area_score * 0.2

        # 3. completeness weight (20%)
        # More contour points may indicate a more complete shape
        contour_points = len(contour)
        if contour_points >= 8:  # At least 8 points are needed to form a good circle
            completeness_score = min(contour_points / 20.0, 1.0)  # 20 points for full score
        else:
            completeness_score = contour_points / 8.0
        confidence += completeness_score * 0.2

        # 4. radius weight (10%)
        # radius should be within a reasonable range
        if 5 <= radius <= 50:
            radius_score = 1.0
        elif radius < 5:
            radius_score = radius / 5.0
        else:
            radius_score = 50.0 / radius
        confidence += radius_score * 0.1

        # 5. area and radius consistency weight (10%)
        # Check the consistency between actual area and theoretical circle area
        theoretical_area = np.pi * radius * radius
        if theoretical_area > 0:
            area_consistency = min(area / theoretical_area, theoretical_area / area)
        else:
            area_consistency = 0
        confidence += area_consistency * 0.1
        
        return min(confidence, 1.0) 
    
    def process_frame(self, ir_raw_image):
        """
        Process a single frame to detect and track markers.
        
        Parameters:
        image (np.ndarray): Input image (should be a binary image after preprocessing)

        Returns:
        tracked_positions (dict): Dictionary of tracked marker positions
        """
        
        results = {}
        # 1. Preprocess the IR image
        # morphological operations to enhance features
        morphological_image = self.morphological_operations(ir_raw_image)
        # high pass filter to extract high-frequency components
        high_freq_image = self.extract_ir_high_freq(morphological_image)
        # Gaussian smoothing to reduce noise
        smooth_image = self.smooth_signal(high_freq_image)
        # Thresholding to create a binary mask
        binary_image = self.threshold_image(smooth_image, threshold_min=50, threshold_max=255)

        # 2. Circle detection
        raw_detections = self.simple_circle_detection(binary_image)

        if not self.tracking_started and len(raw_detections) >= 2:
            # If detected 2 circles, start tracking
            self.marker_tracker.initialize_trackers(raw_detections)
            self.tracking_started = True
            print("Tracking started with initial detections.")
        
        if self.tracking_started:
            # use the tracker to update positions
            tracked_positions = self.marker_tracker.update_trackers(raw_detections)

            # Use the outlier detector to filter outliers
            processed_positions  = self.outlier_detector.process_detections(raw_detections, tracked_positions)

            # Get the check quality of the detections
            detection_quality = self.outlier_detector.get_detection_quality()

            valid_markers = len([p for p in processed_positions.values() if p is not None])
            if valid_markers >= 3:  # at least 3 valid markers to consider tracking successful
                self.successful_tracking_count += 1

            results['tracked_positions'] = tracked_positions
            results['processed_positions'] = processed_positions
            results['detection_quality'] = detection_quality
            results['valid_marker_count'] = valid_markers

        else:
            # If tracking has not started, we can only provide raw detections
            results['tracked_positions'] = {}
            results['processed_positions'] = {}
            results['detection_quality'] = 0
            results['valid_marker_count'] = 0

        # return the results
        results['raw_image'] = ir_raw_image
        results['binary_image'] = binary_image
        results['raw_detections'] = raw_detections
        results['tracking_started'] = self.tracking_started

        return results

    def draw_results(self, results):
        """
        Draw the results on the image.

        Parameters:
        results (dict): Dictionary containing the results from process_frame_with_tracking.
        """
        display_image = cv2.cvtColor(results['binary_image'], cv2.COLOR_GRAY2BGR)

        # Draw tracked positions
        colors = {
            'A': (0, 255, 0),  # Green
            'B': (255, 0, 0),  # Blue
            'C': (0, 0, 255),  # Red
            'D': (255, 255, 0) # Cyan
        }

        for name, pos in results['tracked_positions'].items():
            if pos is not None:
                x, y = pos
                cv2.circle(display_image, (int(x), int(y)), 4, colors[name], -1)
                cv2.putText(display_image, name, (int(x) + 8, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[name], 2)

                if name in results['detection_quality']:
                    quality = results['detection_quality'][name]
                    cv2.putText(display_image, f"Q: {quality:.2f}", (int(x) + 8, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)

        # Draw the text information
        info_text = [
            f"Frame: {self.frame_count}",
            f"Tracking: {'ON' if results['tracking_started'] else 'OFF'}",
            f"Valid Markers: {results['valid_marker_count']}/4",
            f"Raw Detections: {len(results['raw_detections'])}"
        ]

        if self.frame_count > 0:
            success_rate = self.successful_tracking_count / self.frame_count
            info_text.append(f"Success Rate: {success_rate:.2f}")
        
        for i, text in enumerate(info_text):
            cv2.putText(display_image, text, (10, 20 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_image

    def reset_tracking(self):
        """reset the marker tracker and outlier detector"""
        self.marker_tracker.reset()
        self.outlier_detector.reset()
        self.tracking_started = False
        self.successful_tracking_count = 0
        print("Tracking reset.")

    def start(self):
        '''
        Start the RealSense pipeline.
        '''
        try:
            self.pipeline.start(self.config)
            print("Pipeline started successfully.")
        except Exception as e:
            print(f"Error starting pipeline: {e}")
            return False

    def run(self):
        '''
        Run the RealSense pipeline and process frames.
        '''
        try:
            self.start()
            while True:
                frames = self.pipeline.wait_for_frames()
                ir_frame = frames.get_infrared_frame(1)  # Get the second IR frame
                if not ir_frame:
                    continue
                
                # Raw IR image data
                ir_image = np.asanyarray(ir_frame.get_data())
                
                # Process the IR image to detect and track markers
                results = self.process_frame(ir_image)

                display_image = self.draw_results(results)

                # Show result
                cv2.imshow("Result", display_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.reset_tracking()

        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    test = MarkerInit(debug=False)
    test.run()

