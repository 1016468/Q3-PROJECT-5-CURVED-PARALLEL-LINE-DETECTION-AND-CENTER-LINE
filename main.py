"""
Program for live video processing that detects two parallel curves in a defined ROI,
computes the midline between them, and applies smoothing using both an
exponential moving average and a Kalman filter. The resulting smoothed midline is
displayed along with the original curves and processed edges.
"""

import cv2
import numpy as np

# ---------------------------
# ROI Settings
# ---------------------------
ROI_TOP = 210
ROI_BOTTOM = 510
ROI_LEFT = 490
ROI_RIGHT = 790

# Global smoothing parameters
smoothed_midpoints = None
SMOOTHING_ALPHA = 0.05  # Lower value gives a smoother but slower response in this case 0.05 is used

# Fixed number of midpoints for consistent smoothing
NUM_POINTS = 100

# Declares variable to store Kalman filters (one per midline point)
kalman_filters = None


def compute_midline(curve1, curve2):
    """
    Compute the midline between two curves.

    The function assumes that both curves are provided as arrays of points gotten from
    another function and uses the corresponding points from each curve to compute the midpoints.

    Arguments:
        curve1 (np.ndarray): An array of points for the first curve.
        curve2 (np.ndarray): An array of points for the second curve.

    Returns:
        np.ndarray: An array of midpoints computed as the integer average of the
                    corresponding points from curve1 and curve2.
    """
    # Ensure both curves have the same number of points by truncating the longer one.
    min_len = min(len(curve1), len(curve2))
    curve1, curve2 = curve1[:min_len], curve2[:min_len]

    # Compute midpoints using integer division.
    midpoints = (curve1 + curve2) // 2
    return midpoints


def resample_midpoints(midpoints, num_points=NUM_POINTS):
    """
    Resample the array of midpoints to a fixed number of points using linear interpolation.

    This ensures that the smoothing filters always work on an array of a consistent length.

    Arguments:
        midpoints (np.ndarray): The original array of midpoints.
        num_points (int): The desired number of points in the resampled array.

    Returns:
        np.ndarray: The resampled array of midpoints with shape (num_points, 2).
    """
    midpoints = midpoints.astype(np.float32)
    if len(midpoints) == num_points:
        return midpoints

    # Generate indices for linear interpolation.
    indices = np.linspace(0, len(midpoints) - 1, num_points)
    resampled = np.empty((num_points, 2), dtype=np.float32)
    for i, idx in enumerate(indices):
        low = int(np.floor(idx))
        high = min(int(np.ceil(idx)), len(midpoints) - 1)
        weight = idx - low
        resampled[i] = (1 - weight) * midpoints[low] + weight * midpoints[high]
    return resampled


def create_kalman_filter():
    """
    Create and configure a Kalman filter for tracking a 2D point.

    The state vector consists of [x, y, dx, dy] and the measurement vector is [x, y].

    Returns:
        cv2.KalmanFilter: An initialized Kalman filter for 2D point tracking.
    """
    # Initialize Kalman filter with 4 dynamic parameters and 2 measured parameters.
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


def kalman_smooth_midpoints(midpoints):
    """
    Apply Kalman filter smoothing to an array of 2D midpoints.

    One Kalman filter is maintained per point. If the number of points changes,
    the Kalman filters are reinitialized to ensure smooth movement and no flickers.

    Arguments:
        midpoints (np.ndarray): The array of midpoints (shape: [NUM_POINTS, 2]).

    Returns:
        np.ndarray: The array of Kalman-filtered (smoothed) midpoints.
    """
    global kalman_filters

    # Initialize Kalman filters if not already created or if the count has changed.
    if kalman_filters is None or len(kalman_filters) != len(midpoints):
        kalman_filters = [create_kalman_filter() for _ in range(len(midpoints))]
        for i, point in enumerate(midpoints):
            # Initialize the state with the current measurement and zero velocity.
            kalman_filters[i].statePre = np.array([[point[0]], [point[1]], [0], [0]], np.float32)
            kalman_filters[i].statePost = np.array([[point[0]], [point[1]], [0], [0]], np.float32)

    smoothed = []
    for i, point in enumerate(midpoints):
        # Prepare the measurement for the current point.
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        # Predict the next state.
        kalman_filters[i].predict()
        # Correct with the measurement and obtain the estimated state.
        estimated = kalman_filters[i].correct(measurement)
        smoothed.append([estimated[0, 0], estimated[1, 0]])
    return np.array(smoothed, dtype=np.float32)


def detect_curves(frame):
    """
    Detect two parallel curves within the defined ROI of the frame.

    The function performs the following steps:
      1. Extracts the ROI from the frame.
      2. Converts the ROI to grayscale and applies Gaussian blur with a specified kernel size.
      3. Uses Canny edge detection and morphological closing to obtain a clean edge image.
      4. Finds contours in the edge image and selects the two largest as the curves.
      5. Computes the midline between the two curves.
      6. Adjusts the coordinates from the ROI to the full frame.

    Arguments:
        frame (np.ndarray): The full frame captured from the video stream.

    Returns:
        tuple:
            - curve1 (np.ndarray): Points for the first curve.
            - curve2 (np.ndarray): Points for the second curve.
            - midpoints (np.ndarray): The computed midline between curve1 and curve2.
            - edges (np.ndarray): The processed edge image.
    """
    # Extract the region of interest.
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 200)

    # Apply morphological closing to clean up the edges.
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the edge image.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all detected contours on the ROI for debugging purposes.
    cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)

    # Select the two largest contours (assumed to be the parallel curves).
    valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    if len(valid_contours) < 2:
        return None, None, None, edges  # Not enough contours detected.

    # Approximate the contours to reduce noise.
    curve1 = cv2.approxPolyDP(valid_contours[0], epsilon=5, closed=False).reshape(-1, 2)
    curve2 = cv2.approxPolyDP(valid_contours[1], epsilon=5, closed=False).reshape(-1, 2)

    # Compute the midline from the two curves.
    midpoints = compute_midline(curve1, curve2)

    # Adjust the coordinates from ROI to full frame coordinates.
    curve1 += [ROI_LEFT, ROI_TOP]
    curve2 += [ROI_LEFT, ROI_TOP]
    midpoints += [ROI_LEFT, ROI_TOP]

    return curve1, curve2, midpoints, edges


def draw_centerline_on_frame(frame, curve1, curve2, midpoints, edges):
    """
    Draw the detected curves, midline, and processed edges on the frame.

    Args:
        frame (np.ndarray): The original video frame.
        curve1 (np.ndarray): Points of the first curve (drawn in red).
        curve2 (np.ndarray): Points of the second curve (drawn in green).
        midpoints (np.ndarray): Points of the midline (drawn as a blue line).
        edges (np.ndarray): The processed edge image to overlay on the ROI.

    Returns:
        np.ndarray: The frame with all overlays drawn.
    """
    # Draw the first curve in red.
    if curve1 is not None:
        for point in curve1:
            cv2.circle(frame, tuple(map(int, point)), 1, (0, 0, 255), -1)

    # Draw the second curve in green.
    if curve2 is not None:
        for point in curve2:
            cv2.circle(frame, tuple(map(int, point)), 1, (0, 255, 0), -1)

    # Draw the midline (centerline) in blue.
    if midpoints is not None:
        for i in range(len(midpoints) - 1):
            pt1 = tuple(map(int, midpoints[i]))
            pt2 = tuple(map(int, midpoints[i + 1]))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)

    # Overlay the edge image on the ROI.
    if edges is not None:
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = cv2.addWeighted(
            frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT], 0.7, edges_colored, 0.3, 0
        )

    return frame


def main():
    """
    Capture live video, process each frame to detect curves and the midline,
    apply smoothing, and display the results in real time.

    The function continues until the user presses the 'q' key.
    """
    global smoothed_midpoints
    cap = cv2.VideoCapture(0)  # Open the default camera.

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect curves and compute the midline.
        curve1, curve2, midpoints, edges = detect_curves(frame)
        if midpoints is not None:
            # Convert midpoints to float for smoothing.
            current_midpoints = midpoints.astype(np.float32)
            # Resample to a fixed number of points.
            current_midpoints = resample_midpoints(current_midpoints, num_points=NUM_POINTS)

            if smoothed_midpoints is None or len(smoothed_midpoints) != len(current_midpoints):
                smoothed_midpoints = current_midpoints
            else:
                # Apply exponential moving average smoothing.
                smoothed_midpoints = (SMOOTHING_ALPHA * current_midpoints +
                                      (1 - SMOOTHING_ALPHA) * smoothed_midpoints)
            # Further smooth the midpoints using a Kalman filter.
            smoothed_midpoints = kalman_smooth_midpoints(smoothed_midpoints)

            # Draw the original curves and the smoothed centerline.
            frame = draw_centerline_on_frame(frame, curve1, curve2,
                                             smoothed_midpoints.astype(np.int32), edges)

        # Draw the ROI rectangle for visualization.
        cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 255), 2)

        # Display the frame with overlays.
        cv2.imshow("Live Centerline Detection", frame)

        # Exit the loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
