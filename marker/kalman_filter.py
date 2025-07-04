import numpy as np
import cv2

class KalmanFilter2D:
    """
    2D kalman filter for tracking objects in 2D space.
    This filter tracks the position and velocity of an object in 2D space.
    """
    
    def __init__(self, initial_pos, process_noise=1e-2, measurement_noise=1e-1):
        """
        Initialize the Kalman filter with initial position and noise parameters.
        
        Args:
            initial_pos: Initial position (x, y)
            process_noise: Process noise
            measurement_noise: Measurement noise
        """
        self.kalman = cv2.KalmanFilter(4, 2)

        # State transition matrix (position = position + velocity*dt, dt=1)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (can only observe position, not velocity)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance matrix
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance matrix
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Posterior error covariance matrix
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        # Initial state [x, y, vx, vy]
        self.kalman.statePre = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        
        self.last_measurement = np.array(initial_pos, dtype=np.float32)
        self.prediction_count = 0  # Continuous prediction count
        self.max_prediction_frames = 10  # Maximum continuous prediction frames

    def predict(self):
        """Predict the next state"""
        prediction = self.kalman.predict()
        return prediction[0], prediction[1]
    
    def update(self, measurement):
        """Update the filter with the measurement"""
        if measurement is not None:
            measurement_array = np.array([measurement[0], measurement[1]], dtype=np.float32)
            self.kalman.correct(measurement_array)
            self.last_measurement = measurement_array
            self.prediction_count = 0
        else:
            # If no measurement is available, increment prediction count
            self.prediction_count += 1

        # If continuous prediction count exceeds limit, reset to last measurement
        if self.prediction_count > self.max_prediction_frames:
            self.kalman.statePost = np.array([
                self.last_measurement[0], self.last_measurement[1], 0, 0
            ], dtype=np.float32)
            self.prediction_count = 0
    
    def get_state(self):
        """Get the current position"""
        return self.kalman.statePost[0], self.kalman.statePost[1]
    
    def get_velocity(self):
        """Get the current velocity"""
        return self.kalman.statePost[2], self.kalman.statePost[3]


class MarkerTracker:
    """
    Multi-marker tracker, managing multiple Kalman filters.
    """
    
    def __init__(self, marker_names=['A', 'B', 'C', 'D']):
        self.marker_names = marker_names
        self.trackers = {}
        self.is_initialized = False
        self.match_threshold = 50  # Matching threshold (pixels)
        self.confidence_threshold = 0.3  # Confidence threshold

    def initialize_trackers(self, detections):
        """
        Initialize trackers

        Args:
            detections: Detection results list [(x, y, radius, confidence), ...]
        """
        if len(detections) >= len(self.marker_names):
            # Sort by confidence
            detections.sort(key=lambda x: x[3], reverse=True)
            
            for i, name in enumerate(self.marker_names):
                x, y, _, _ = detections[i]
                self.trackers[name] = KalmanFilter2D((x, y))
            
            self.is_initialized = True
            print(f"Initialized {len(self.trackers)} trackers")

    def update_trackers(self, detections):
        """
        Update trackers

        Args:
            detections: Detection results list [(x, y, radius, confidence), ...]
        """
        if not self.is_initialized:
            # If not initialized, try to initialize
            if len(detections) >= len(self.marker_names):
                self.initialize_trackers(detections)
            return {}

        # Predict all tracker positions
        predictions = {}
        for name, tracker in self.trackers.items():
            pred_x, pred_y = tracker.predict()
            predictions[name] = (pred_x, pred_y)

        # Match detections to trackers
        matched_detections = self._match_detections_to_trackers(detections, predictions)

        # Update trackers
        updated_positions = {}
        for name, tracker in self.trackers.items():
            if name in matched_detections:
                detection = matched_detections[name]
                tracker.update((detection[0], detection[1]))
            else:
                # If no matching detection, use prediction
                tracker.update(None)

            # Get updated position
            x, y = tracker.get_state()
            updated_positions[name] = (x, y)
        
        return updated_positions
    
    def _match_detections_to_trackers(self, detections, predictions):
        """
        Match detection results to trackers

        Args:
            detections: Detection results list
            predictions: Prediction positions dictionary

        Returns:
            Matching results dictionary
        """
        matched = {}
        used_detections = set()
        
        # filter detections by confidence threshold
        valid_detections = [d for d in detections if d[3] > self.confidence_threshold]
        
        # for each prediction, find the best matching detection
        for name, (pred_x, pred_y) in predictions.items():
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            for i, (x, y, radius, confidence) in enumerate(valid_detections):
                if i in used_detections:
                    continue
                
                # calculate distance from prediction to detection
                distance = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
                
                if distance < self.match_threshold and distance < best_distance:
                    best_match = (x, y, radius, confidence)
                    best_distance = distance
                    best_idx = i
            
            if best_match is not None:
                matched[name] = best_match
                used_detections.add(best_idx)
        
        return matched
    
    def get_marker_positions(self):
        """Get the current positions of all markers"""
        positions = {}
        for name, tracker in self.trackers.items():
            x, y = tracker.get_state()
            positions[name] = (x, y)
        return positions
    
    def reset(self):
        """reset the tracker"""
        self.trackers = {}
        self.is_initialized = False