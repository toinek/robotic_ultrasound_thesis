import cv2
import numpy as np
import rospy
import pyrealsense2 as rs

from geometry_msgs.msg import PoseArray, Pose
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev

from realsense_mediapipe import RealSenseCameraMediaPipe

class BlueLineDetector(RealSenseCameraMediaPipe):
    def __init__(self, **kwargs):
        kwargs['clip'] = 1.5
        super().__init__(**kwargs)
        self.blue_line_pub = rospy.Publisher('detection/blue_line', PoseArray, queue_size=10)
        self.create_trackbar_window()  # Create the trackbars for HSV tuning

    def create_trackbar_window(self):
        """Creates an OpenCV window with trackbars for tuning HSV thresholds."""
        cv2.namedWindow("HSV Trackbars")
        cv2.createTrackbar("Low H", "HSV Trackbars", 95, 180, lambda x: None)
        cv2.createTrackbar("High H", "HSV Trackbars", 135, 180, lambda x: None)
        cv2.createTrackbar("Low S", "HSV Trackbars", 180, 255, lambda x: None)
        cv2.createTrackbar("High S", "HSV Trackbars", 255, 255, lambda x: None)
        cv2.createTrackbar("Low V", "HSV Trackbars", 70, 255, lambda x: None)
        cv2.createTrackbar("High V", "HSV Trackbars", 255, 255, lambda x: None)

    def get_hsv_values(self):
        """Retrieves the current values of the HSV trackbars."""
        l_h = cv2.getTrackbarPos("Low H", "HSV Trackbars")
        h_h = cv2.getTrackbarPos("High H", "HSV Trackbars")
        l_s = cv2.getTrackbarPos("Low S", "HSV Trackbars")
        h_s = cv2.getTrackbarPos("High S", "HSV Trackbars")
        l_v = cv2.getTrackbarPos("Low V", "HSV Trackbars")
        h_v = cv2.getTrackbarPos("High V", "HSV Trackbars")
        return np.array([l_h, l_s, l_v]), np.array([h_h, h_s, h_v])

    def apply_clahe(self, image):
        """Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def extract_centerline(self, depth_mask, depth_frame, color_image):
        # Step 1: Skeletonization to extract the centerline
        skeleton = skeletonize(depth_mask > 0).astype(np.uint8) * 255  # Convert to binary

        # Step 2: Extract centerline points
        centerline_points = np.column_stack(np.where(skeleton > 0))

        smooth_centerline_3d = []  # Store 3D coordinates

        if len(centerline_points) > 10:  # Ensure enough points for smoothing
            # Step 3: Fit a smooth curve using spline interpolation
            centerline_points = centerline_points[:, ::-1]  # Convert (row, col) to (x, y)
            tck, u = splprep(centerline_points.T, s=100)  # s=5 controls smoothness
            smooth_centerline = np.array(splev(np.linspace(0, 1, 100), tck)).T.astype(int)

            prev_point = None  # To store the previous point for drawing
            for pixel in smooth_centerline:
                u, v = pixel  # Pixel coordinates
                try:
                    depth = depth_frame.get_distance(u, v)  # Get depth value at pixel
                except:
                    continue
                if 0 < depth < self.clip_distance:  # Ensure valid depth
                    point_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u, v], depth)
                    smooth_centerline_3d.append(point_3d)

                # Draw the smooth centerline on the color image
                if prev_point is not None:
                    cv2.line(color_image, tuple(prev_point), tuple(pixel), (0, 0, 255), 2)
                prev_point = pixel

        return smooth_centerline_3d

    def detect_blue_line(self, color_image, depth_frame):
        """ Detects a blue line using adaptive HSV thresholding and extracts its centerline """
        enhanced_image = self.apply_clahe(color_image)
        hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)

        # Get HSV values dynamically from the trackbars
        lower_blue, upper_blue = self.get_hsv_values()
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Apply morphological operations to fill gaps in the line
        kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Reduce small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=5)  # Fill gaps

        # Find contours and filter by area
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        valid_contours = []
        depth_mask = np.zeros_like(mask, dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Keep only large contours
                if any(0 < depth_frame.get_distance(u, v) < self.clip_distance for point in cnt for u, v in [point[0]]):
                    valid_contours.append(cnt)

        if valid_contours == []:
            return None, None

        # Draw and fill only the valid contours
        cv2.drawContours(depth_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

        # smooth_centerline_3d = self.extract_centerline(depth_mask, depth_frame, color_image)

        return valid_contours, depth_mask

    def visualize_blue_line(self, color_image, mask):
        """ Visualizes the detected blue line regions """
        mask_visual = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((color_image, mask_visual))
        cv2.imshow('blue Line Detection', combined)
        cv2.waitKey(1)

    def publish_blue_line(self, centerline_points):
        """ Publishes the detected blue line centerline as a PoseArray message """
        pose_array = PoseArray()
        pose_array.header.frame_id = "world"
        pose_array.header.stamp = rospy.Time.now()

        for point in centerline_points:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = point
            pose_array.poses.append(pose)

        self.blue_line_pub.publish(pose_array)

    def _run(self):
        """ Detects the blue line and overlays the smoothed centerline in 2D and 3D. """
        keypoints = super()._run(show_img=False)

        # color_frame, depth_frame = self.get_frames()
        color_frame, depth_frame = self.color_frame, self.depth_frame
        color_image = np.asanyarray(color_frame.get_data())

        if keypoints is None:
            cv2.imshow('blue Line Detection', color_image)
            cv2.waitKey(1)
            return

        valid_contours, mask = self.detect_blue_line(color_image, depth_frame)
        if valid_contours:
            blue_line_3d = self.extract_centerline(mask, depth_frame, self.image_with_trajectory)

            self.visualize_blue_line(self.image_with_trajectory, mask)
            aligned_line = self.align_pcd(blue_line_3d, publish=False)
            if aligned_line is not None:
                self.publish_blue_line(aligned_line)



if __name__ == '__main__':
    rospy.init_node('blue_line_detection')
    detector = BlueLineDetector(debug=False)

    while not rospy.is_shutdown():
        detector._run()
