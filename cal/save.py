import os
import pyrealsense2 as rs
import cv2
import numpy as np

# -------------------------
# 相机流参数
# -------------------------
SensorName = "RGB"

if SensorName == "IR":
    IR_INDEX = 1  

WIDTH, HEIGHT = 1920, 1080  # 分辨率
FREQ = 30         # 帧率

# -------------------------
# 配置保存目录
# -------------------------
if SensorName == "IR":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, f'ir{IR_INDEX}_calib_images')
    os.makedirs(save_dir, exist_ok=True)
elif SensorName == "RGB":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, 'rgb_calib_images')
    os.makedirs(save_dir, exist_ok=True)

print(f"保存目录: {save_dir}")

# -------------------------
# 启动 RealSense 管线
# -------------------------
pipeline = rs.pipeline()
cfg = rs.config()
if SensorName == "IR":
    cfg.enable_stream(rs.stream.infrared, IR_INDEX, WIDTH, HEIGHT, rs.format.y8, FREQ)
elif SensorName == "RGB":
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FREQ)
profile = pipeline.start(cfg)

# -------------------------
# 配置 IR / RGB 摄像头
# -------------------------
for sensor in profile.get_device().sensors:
    name = sensor.get_info(rs.camera_info.name).lower()
    # 立体模块里的 IR 摄像头
    if 'stereo' in name:
        print(f"配置传感器: {name}")

        # 关闭自动曝光
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            print("  已关闭自动曝光")

        # 固定曝光时间（微秒）
        if sensor.supports(rs.option.exposure):
            sensor.set_option(rs.option.exposure, 1000)
            print("  曝光设置为 1000us")

        # 查询并设置增益
        if sensor.supports(rs.option.gain):
            rng = sensor.get_option_range(rs.option.gain)
            print(f"  Gain 支持范围: {rng.min} – {rng.max} (step {rng.step})")
            # 取区间中点
            mid_gain = rng.min + (rng.max - rng.min) / 2
            # 对齐到步长
            mid_gain = rng.min + round((mid_gain - rng.min) / rng.step) * rng.step
            sensor.set_option(rs.option.gain, mid_gain)
            print(f"  已将 Gain 设置为 {mid_gain}")

        # 查询并设置 IR 投射强度
        # if sensor.supports(rs.option.laser_power):
        #     rng2 = sensor.get_option_range(rs.option.laser_power)
        #     print(f"  Laser Power 范围: {rng2.min} – {rng2.max}")
        #     # 取最大值的 80%
        #     power = rng2.min + (rng2.max - rng2.min) * 0.8
        #     sensor.set_option(rs.option.laser_power, power)
        #     print(f"  已将 Laser Power 设置为 {power}")
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 0)
            print("  已关闭 IR 投射")

# -------------------------
# 实时显示并手动确认保存
# -------------------------
if SensorName == "IR":
    cv2.namedWindow('IR Preview', cv2.WINDOW_NORMAL)
elif SensorName == "RGB":
    cv2.namedWindow('RGB Preview', cv2.WINDOW_NORMAL)

img_count = 0   
print("实时流: 按 's' 冻结并进入确认，按 'q' 退出。")

try:
    while True:
        # 1ms 刷新一次流
        frames = pipeline.wait_for_frames()
        # 获取 IR 图像\
        if SensorName == "IR":
            ir_frame = frames.get_infrared_frame(IR_INDEX)
            img = np.asanyarray(ir_frame.get_data())
            cv2.imshow('IR Preview', img)
        elif SensorName == "RGB":
            color_frame = frames.get_color_frame()
            img = np.asanyarray(color_frame.get_data())
            cv2.imshow('RGB Preview', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # 冻结当前帧，进入确认模式
            print(f"已冻结帧 {img_count:03d}，按 'y' 保存、'n' 丢弃、或 'q' 退出。")
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('y'):
                    if SensorName == "IR":
                        filename = f'ir_{img_count:03d}.png'
                    elif SensorName == "RGB":
                        filename = f'rgb_{img_count:03d}.png'
                    filepath = os.path.join(save_dir, filename)
                    cv2.imwrite(filepath, img)
                    print(f"已保存: {filepath}")
                    img_count += 1
                    break
                elif k2 == ord('n'):
                    print("已丢弃该帧，不保存。")
                    break
                elif k2 == ord('q'):
                    print("退出采集。")
                    raise KeyboardInterrupt
                else:
                    continue
            print("恢复实时流，继续按 's' 冻结下一帧 或 'q' 退出。")

        elif key == ord('q'):
            print("退出采集。")
            break

finally:
    # 停止管线并关闭窗口
    pipeline.stop()
    cv2.destroyAllWindows()
