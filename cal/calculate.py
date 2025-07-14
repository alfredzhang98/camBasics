import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SensorName = "RGB"

if SensorName == "IR":
    IR_INDEX = 1

# -------------------------
# 路径配置
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
if SensorName == "IR":
    save_dir = os.path.join(base_dir, f'{SensorName}{IR_INDEX}_calib_images')
    vis_dir = os.path.join(base_dir, f'{SensorName}{IR_INDEX}_calib_vis')
    os.makedirs(vis_dir, exist_ok=True)
elif SensorName == "RGB":
    save_dir = os.path.join(base_dir, f'{SensorName}_calib_images')
    vis_dir = os.path.join(base_dir, f'{SensorName}_calib_vis')
    os.makedirs(vis_dir, exist_ok=True)

# -------------------------
# 棋盘格参数
# -------------------------
# 横向 6 黑 + 6 白 = 12 格 => 内角点数 11
# 纵向 5 黑 + 4 白 = 9 格  => 内角点数 8
pattern_size = (11, 8)      # (宽, 高)
square_size_mm = 15         # 每格边长（毫米）

# -------------------------
# 准备世界坐标点
# -------------------------
objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size_mm

# -------------------------
# 查找角点并收集点对
# -------------------------
objpoints = []  # 3D 点
imgpoints = []  # 2D 点
if SensorName == "IR":
    image_paths = sorted(glob.glob(os.path.join(save_dir, 'ir_*.png')))
elif SensorName == "RGB":
    image_paths = sorted(glob.glob(os.path.join(save_dir, 'rgb_*.png')))

for fname in image_paths:
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取: {fname}")
        continue
    ret, corners = cv2.findChessboardCorners(img, pattern_size)
    if not ret:
        print(f"未检测到角点: {fname}")
        continue
    corners_sub = cv2.cornerSubPix(
        img, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    objpoints.append(objp)
    imgpoints.append(corners_sub)
    print(f"角点已添加: {os.path.basename(fname)}")

# -------------------------
# 相机标定
# -------------------------
if len(objpoints) < 5:
    raise RuntimeError("至少需要5张有效图像进行标定！")

# 标定
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img.shape[::-1], None, None
)

print("标定完成:")
print("相机矩阵 K:\n", K)
print("畸变系数 dist:\n", dist.ravel())

# 保存内参
calib_data = {
    'camera_matrix': K.tolist(),
    'distortion_coefficients': dist.ravel().tolist(),
    'pattern_size': pattern_size,
    'square_size_mm': square_size_mm,
}
if SensorName == "IR":
    with open(os.path.join(base_dir, f'ir{IR_INDEX}_intrinsics.json'), 'w') as f:
        json.dump(calib_data, f, indent=4)
    print("内参已保存至 ir{IR_INDEX}_intrinsics.json")
elif SensorName == "RGB":
    with open(os.path.join(base_dir, 'rgb_intrinsics.json'), 'w') as f:
        json.dump(calib_data, f, indent=4)
    print("内参已保存至 rgb_intrinsics.json")

# -------------------------
# 可视化 1: 绘制角点
# -------------------------
for i, fname in enumerate(image_paths):
    img_gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img_gray is None or i >= len(imgpoints):
        continue
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(img_color, pattern_size, imgpoints[i], True)
    out_file = os.path.join(vis_dir, f'corners_{i:03d}.png')
    cv2.imwrite(out_file, img_color)
print(f"角点可视化已保存至 {vis_dir}")

# -------------------------
# 可视化 2: 重投影误差曲线
# -------------------------
errors = []
for objp_i, imgp_i, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
    proj, _ = cv2.projectPoints(objp_i, rvec, tvec, K, dist)
    err = cv2.norm(imgp_i, proj, cv2.NORM_L2) / len(proj)
    errors.append(err)

plt.figure()
plt.plot(errors, marker='o')
plt.title('Reprojection Error per Image')
plt.xlabel('Image Index')
plt.ylabel('Error (pixels)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'reprojection_error.png'))
plt.close()
print("重投影误差图已保存至 reprojection_error.png")

# -------------------------
# 可视化 3: 畸变校正对比
# -------------------------
img0 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
h, w = img0.shape[:2]
mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
undist = cv2.remap(img0, mapx, mapy, cv2.INTER_LINEAR)
both = np.hstack((img0, undist))
out = os.path.join(vis_dir, 'undistort_compare.png')
cv2.imwrite(out, both)
print("畸变前后对比图已保存至 undistort_compare.png")


# -------------------------
# 可视化 4: 交互式 3D 窗口展示相机位姿
# -------------------------
plt.ion()  # 打开交互模式
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 绘制棋盘格世界点
ax.scatter(objp[:,0], objp[:,1], objp[:,2], c='r', marker='o', label='Chessboard Points')
# 绘制每个相机位姿
for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec.reshape(3,)
    ax.scatter(C[0], C[1], C[2], c='b', marker='^')
    # 相机坐标系轴
    axes = np.eye(3) * square_size_mm * 2
    for j, axis in enumerate(['X','Y','Z']):
        dir_vec = R.T @ axes[:, j]
        ax.plot([C[0], C[0]+dir_vec[0]],
                [C[1], C[1]+dir_vec[1]],
                [C[2], C[2]+dir_vec[2]],
                label=f'Cam{i}_{axis}' if j==0 else None)
# 设置视角及标签
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Interactive Camera Poses')
# 关闭legend
ax.legend().set_visible(False)
plt.show()
# 保持窗口打开直到手动关闭
input("按回车键关闭 3D 可视化窗口...")
