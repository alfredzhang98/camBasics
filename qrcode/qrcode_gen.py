import cv2
import cv2.aruco as aruco

# 选择字典（常用 DICT_4X4_50）
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker_size_pixels = 600

# 生成第一个 marker（ID = 23）
marker_id1 = 23
img1 = aruco.generateImageMarker(aruco_dict, marker_id1, marker_size_pixels)
cv2.imwrite("aruco_marker_23.png", img1)

# 生成第二个 marker（ID = 45）
marker_id2 = 45
img2 = aruco.generateImageMarker(aruco_dict, marker_id2, marker_size_pixels)
cv2.imwrite("aruco_marker_45.png", img2)

print("Markers saved!")
