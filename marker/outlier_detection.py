import numpy as np
from collections import deque
import cv2

class OutlierDetector:
    """
    Outlier detection class for tracking positions
    This class uses velocity and acceleration thresholds to detect outliers in position data.
    """
    
    def __init__(self, history_size=5, velocity_threshold=200, acceleration_threshold=100):
        """
        Initialize the outlier detector

        Args:
            history_size: Size of the history buffer
            velocity_threshold: Velocity outlier threshold (pixels/frame)
            acceleration_threshold: Acceleration outlier threshold (pixels/frame²)
        """
        self.history_size = history_size
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold

        # Position history buffer
        self.position_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size-1)
        self.acceleration_history = deque(maxlen=history_size-2)

        # Statistics
        self.mean_velocity = 0
        self.std_velocity = 0
        self.mean_acceleration = 0
        self.std_acceleration = 0
        
    def add_position(self, position):
        """
        Add a new position and update statistics

        Args:
            position: (x, y) position coordinates
        """
        self.position_history.append(position)

        # Calculate velocity
        if len(self.position_history) >= 2:
            prev_pos = self.position_history[-2]
            curr_pos = self.position_history[-1]
            velocity = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            self.velocity_history.append(velocity)

            # Update velocity statistics
            if len(self.velocity_history) > 3:
                velocities = list(self.velocity_history)
                self.mean_velocity = np.mean(velocities)
                self.std_velocity = np.std(velocities)

        # Calculate acceleration
        if len(self.velocity_history) >= 2:
            prev_vel = self.velocity_history[-2]
            curr_vel = self.velocity_history[-1]
            acceleration = abs(curr_vel - prev_vel)
            self.acceleration_history.append(acceleration)

            # Update acceleration statistics
            if len(self.acceleration_history) > 3:
                accelerations = list(self.acceleration_history)
                self.mean_acceleration = np.mean(accelerations)
                self.std_acceleration = np.std(accelerations)
    
    def is_outlier(self, new_position):
        """
        Check if the new position is an outlier

        Args:
            new_position: (x, y) position coordinates

        Returns:
            bool: True if outlier, False if normal
        """
        if len(self.position_history) < 2:
            return False

        # Calculate the velocity of the new position relative to the last position
        last_pos = self.position_history[-1]
        current_velocity = np.sqrt((new_position[0] - last_pos[0])**2 + (new_position[1] - last_pos[1])**2)

        # Velocity outlier detection
        if current_velocity > self.velocity_threshold:
            return True

        # Historical velocity outlier detection
        if len(self.velocity_history) > 3:
            velocity_z_score = abs(current_velocity - self.mean_velocity) / (self.std_velocity + 1e-6)
            if velocity_z_score > 3:  # 3-sigma rule
                return True

        # Acceleration outlier detection
        if len(self.velocity_history) >= 1:
            last_velocity = self.velocity_history[-1]
            current_acceleration = abs(current_velocity - last_velocity)
            
            if current_acceleration > self.acceleration_threshold:
                return True

            # Historical acceleration outlier detection
            if len(self.acceleration_history) > 3:
                accel_z_score = abs(current_acceleration - self.mean_acceleration) / (self.std_acceleration + 1e-6)
                if accel_z_score > 3:  # 3-sigma rule
                    return True
        
        return False
    
    def get_predicted_position(self):
        """
        Predict the next position based on historical data

        Returns:
            (x, y): Predicted position
        """
        if len(self.position_history) < 2:
            return self.position_history[-1] if self.position_history else (0, 0)

        # Simple linear prediction
        if len(self.position_history) >= 2:
            last_pos = self.position_history[-1]
            second_last_pos = self.position_history[-2]

            # Calculate velocity vector
            vx = last_pos[0] - second_last_pos[0]
            vy = last_pos[1] - second_last_pos[1]

            # Predict the next position
            predicted_x = last_pos[0] + vx
            predicted_y = last_pos[1] + vy
            
            return (predicted_x, predicted_y)
        
        return self.position_history[-1]
    
    def get_smoothed_position(self):
        """
        Get the smoothed position (weighted average based on historical data)

        Returns:
            (x, y): Smoothed position
        """
        if not self.position_history:
            return (0, 0)
        
        if len(self.position_history) == 1:
            return self.position_history[0]
        
        # Use exponential weighting for smoothing
        weights = np.exp(np.linspace(-1, 0, len(self.position_history)))
        weights = weights / np.sum(weights)
        
        positions = list(self.position_history)
        smoothed_x = np.sum([w * pos[0] for w, pos in zip(weights, positions)])
        smoothed_y = np.sum([w * pos[1] for w, pos in zip(weights, positions)])
        
        return (smoothed_x, smoothed_y)


class RobustMarkerDetector:
    """
    Robust marker detector with outlier detection
    """
    
    def __init__(self, marker_names=['A', 'B', 'C', 'D']):
        self.marker_names = marker_names
        self.outlier_detectors = {name: OutlierDetector() for name in marker_names}
        self.valid_detections = {name: None for name in marker_names}
        self.missing_count = {name: 0 for name in marker_names}
        self.max_missing_frames = 5  # Maximum missing frames

    def process_detections(self, detections, tracker_positions):
        """
        Process detection results and filter outliers

        Args:
            detections: Original detection results
            tracker_positions: Tracker position dictionary

        Returns:
            Processed detection results
        """
        processed_detections = {}
        
        for name in self.marker_names:
            if name in tracker_positions:
                current_pos = tracker_positions[name]
                detector = self.outlier_detectors[name]

                # Check for outliers
                if detector.is_outlier(current_pos):
                    # If it's an outlier, use the predicted position
                    predicted_pos = detector.get_predicted_position()
                    processed_detections[name] = predicted_pos
                    self.missing_count[name] += 1

                    print(f"Marker {name} detected an outlier, using predicted position: {predicted_pos}")
                else:
                    # If it's not an outlier, accept the detection result
                    detector.add_position(current_pos)
                    processed_detections[name] = current_pos
                    self.valid_detections[name] = current_pos
                    self.missing_count[name] = 0
            else:
                # Marker not detected, increase missing count
                self.missing_count[name] += 1
                
                if self.missing_count[name] <= self.max_missing_frames:
                    # If missing frames are not many, use predicted position
                    detector = self.outlier_detectors[name]
                    predicted_pos = detector.get_predicted_position()
                    processed_detections[name] = predicted_pos
                    print(f"Marker {name} lost, using predicted position: {predicted_pos}")
                else:
                    # If lost for too long, mark as invalid
                    processed_detections[name] = None
                    print(f"Marker {name} lost for too long, marked as invalid")

        return processed_detections
    
    def get_detection_quality(self):
        """
        Get detection quality assessment
        
        Returns:
            dict: Quality assessment for each marker
        """
        quality = {}
        for name in self.marker_names:
            missing_ratio = self.missing_count[name] / (self.max_missing_frames + 1)
            quality[name] = max(0, 1 - missing_ratio)
        
        return quality
    
    def reset(self):
        """Reset the detector"""
        self.outlier_detectors = {name: OutlierDetector() for name in self.marker_names}
        self.valid_detections = {name: None for name in self.marker_names}
        self.missing_count = {name: 0 for name in self.marker_names}