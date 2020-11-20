"""Project a virtual object (3D) into the scene.

Note: In week2 assignment, we only need to project a logo image (2D).
We can directly estimate the homorgraphy (3 X 3) with 4 points correspondence.
In order to project a 3D object, the homorgraphy is (4 X 4). We need 8 points
correspondence. If the intrinsic parameter K is given, we can estimate the
rotation matrix and translation vector with 4 coplanar points correspondence.
"""


# standard library imports
import os

# related third party imports
import cv2
import numpy as np

# local application/library specific imports
from utils import est_homography, est_Rt

# Initialize the tracker
# The tracker in opencv only supports tracking a bounding box.
# TODO: We need a tracker to accurately track all the points.
tracker = cv2.TrackerTLD_create()
# Four initial points for tracker (row, col).
# The first point is at top-right.
# The remaining points are organized in clockwise order.
tracker_init_pts = np.array([
    [168.6005859375000, 403.6800842285157],
    [342.4402770996094, 378.6268920898438],
    [316.5294189453125, 198.1631469726562],
    [149.1907043457031, 233.3528289794922]
])

# Reference points in world coordinate
tag_width = 0.13
tag_height = 0.13
# Four reference points (row, col). The origin is the center of the tag.
# The first point is at top-right.
# The remaining points are organized in clockwise order.
ref_pts = np.array([
    [tag_height / 2.0, tag_width / 2.0],
    [-tag_height / 2.0, tag_width / 2.0],
    [-tag_height / 2.0, -tag_width / 2.0],
    [tag_height / 2.0, -tag_width / 2.0]
])

# Points to be projected to each frame
cube_depth = 0.13
sample_pts_part1 = np.concatenate((ref_pts, np.zeros((4, 1))), axis=1)
sample_pts_part2 = np.concatenate((ref_pts, cube_depth * np.ones((4, 1))), axis=1)
sample_pts = np.concatenate((sample_pts_part1, sample_pts_part2), axis=0)

# Intrinsic parameters matrix
K = np.array([
    [766.1088867187500, 0, 313.9585628047498],
    [0, 769.9354248046875, 250.3607131410900],
    [0, 0, 1.0]
])
K_inv = np.linalg.inv(K)

# Process all the images
proj_dir = "/mnt/c/research/Coding_Practice/Coursera_Robotics_Perception/RoboticsPerceptionWeek3Code/data/projected_images"
frame_idx = 0
while True:
    # read a new frame and track the reference points
    frame = cv2.imread("data/apriltagims/image{:03d}.jpg".format(frame_idx + 1))
    tracked_pts = tracker_init_pts

    # Transform the tracked points from image coordinate into
    # camera coordinate
    tracked_pts_hm = np.concatenate((tracked_pts, np.ones((4, 1))), axis=1)
    points_c_hm = np.matmul(K_inv, np.transpose(tracked_pts_hm))
    # points_c_hm equals to (x, y, 1).
    points_c = points_c_hm[:2, :]

    # Estimate the homorgraphy that transforms points from word coordinate
    # to camera coordinate
    H = est_homography(ref_pts, np.transpose(points_c))

    # Estimate the rotation and translation of camera
    R, t = est_Rt(H)

    # Transform points from word coordinate to image coordinate
    proj_c = np.matmul(R, np.transpose(sample_pts)) + np.tile(np.expand_dims(t, axis=1), reps=(1, sample_pts.shape[0]))
    proj_pts_hm = np.matmul(K, proj_c)
    proj_pts = proj_pts_hm[:2, :]
    proj_pts[0, :] = proj_pts[0, :] / proj_pts_hm[2, :]
    proj_pts[1, :] = proj_pts[1, :] / proj_pts_hm[2, :]

    # Draw the AR cube in the image
    proj_pts = np.transpose(proj_pts).astype(np.int)
    points_idx_draw = [(0, 1), (1, 2), (2, 3), (3, 0),
                       (4, 5), (5, 6), (6, 7), (7, 4),
                       (0, 4), (1, 5), (2, 6), (3, 7)]
    for line in points_idx_draw:
        frame = cv2.line(
            frame,
            (proj_pts[line[0], 1], proj_pts[line[0], 0]),
            (proj_pts[line[1], 1], proj_pts[line[1], 0]),
            (255, 0, 0),
            2
        )

    cv2.imwrite(os.path.join(proj_dir, "image{:03d}.png".format(frame_idx + 1)), frame)

    frame_idx += 1
    break

print("End!")
