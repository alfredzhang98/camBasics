#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import pyrealsense2 as rs
import sys
from collections import deque


def load_calibration(calib_filename='rgb_intrinsics.json', target_size=(1920, 1080)):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(base_dir, calib_filename)
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with open(calib_path, 'r') as f:
        calib = json.load(f)

    # 原始标定参数
    K_orig = np.array(calib['camera_matrix'], dtype=np.float32)
    dist = np.array(calib['distortion_coefficients'], dtype=np.float32).reshape(-1, 1)

    # 为 target_size 计算新的相机矩阵
    w, h = target_size
    new_K, _ = cv2.getOptimalNewCameraMatrix(K_orig, dist, (w, h), 1)
    return new_K, dist


def main():
    frame_size = (1920, 1080) 
    cam_matrix, dist_coeffs = load_calibration(target_size=frame_size)
    print("Adapted camera matrix:\n", cam_matrix)
    print("Distortion coeffs:\n", dist_coeffs.flatten())

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, 30)
    pipeline.start(config)

    aruco_dict    = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector      = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    # 4x4 ArUco marker size in meters
    # IF it is 3cm marker, then marker_length = 0.03
    marker_length = 0.04

    half = marker_length / 2
    objp = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float32)

    pose_windows = {}
    window_size = 10

    try:
        while True:

            # Key exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("Exiting...")
                break

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(gray)
            corners = list(corners)
            if ids is None or len(ids) == 0:
                cv2.imshow("ArUco 6DoF", img)
                if cv2.waitKey(1) == 27:
                    break
                continue

            for i, c in enumerate(corners):
                pts = c.reshape(-1, 1, 2).astype(np.float32)
                cv2.cornerSubPix(
                    gray, pts,
                    winSize=(5, 5), zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                )
                corners[i] = pts.reshape(1, 4, 2)

            rvecs, tvecs = [], []
            for c in corners:
                image_pts = c.reshape((4, 2)).astype(np.float32)
                undist = cv2.undistortPoints(
                    image_pts.reshape(-1, 1, 2),
                    cam_matrix, dist_coeffs, P=cam_matrix
                ).reshape(-1, 2)

                ok, rvec, tvec = cv2.solvePnP(
                    objp, undist, cam_matrix, None,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                rvecs.append(rvec if ok else None)
                tvecs.append(tvec if ok else None)

            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            # 构建单行输出字符串
            out_str = []
            for idx, marker_id in enumerate(ids.flatten()):
                if rvecs[idx] is None:
                    continue
                t = tvecs[idx].flatten()
                if marker_id not in pose_windows:
                    pose_windows[marker_id] = deque(maxlen=window_size)
                pose_windows[marker_id].append(t)
                avg_t = np.mean(pose_windows[marker_id], axis=0)
                r = rvecs[idx].flatten()
                out_str.append(
                    f"ID={marker_id}: t=[{avg_t[0]:.5f},{avg_t[1]:.5f},{avg_t[2]:.5f}]m"
                    f", r=[{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}]rad"
                )
                cv2.drawFrameAxes(
                    img, cam_matrix, dist_coeffs,
                    rvecs[idx], avg_t.reshape(3, 1), half
                )
            if len(ids) >= 2 and len(pose_windows)>=2:
                # 取最新两标记平滑后位置
                ids_list = list(ids.flatten())
                t1 = pose_windows[ids_list[0]][-1]
                t2 = pose_windows[ids_list[1]][-1]
                dt = t2 - t1
                dist_xyz = np.linalg.norm(dt)
                out_str.append(f"DistXYZ={dist_xyz:.5f}m")
                # add the rotation difference
                r1 = rvecs[0].flatten()
                r2 = rvecs[1].flatten()
                r_diff = np.linalg.norm(r2 - r1)
                out_str.append(f"RotDiff={r_diff:.5f}rad")

                # 1) 计算两个标记的像素中心
                pts1 = corners[0].reshape(4, 2)
                pts2 = corners[1].reshape(4, 2)
                c1 = tuple(np.mean(pts1, axis=0).astype(int))
                c2 = tuple(np.mean(pts2, axis=0).astype(int))

                # 2) 画线和两个小圆点
                cv2.line(img, c1, c2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(img, c1, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(img, c2, 5, (0, 0, 255), -1, cv2.LINE_AA)

                # 3) 在连线中点处标注距离值
                mid_pt = ((c1[0]+c2[0])//2, (c1[1]+c2[1])//2)
                cv2.putText(
                    img, f"{dist_xyz*100:.1f}cm", mid_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

            # 输出到同一行并回退
            sys.stdout.write(" | ".join(out_str) + "\r")
            sys.stdout.flush()

            cv2.imshow("ArUco 6DoF", img)
            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
