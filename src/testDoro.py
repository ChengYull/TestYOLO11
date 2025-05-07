import cv2
# 导入YOLO模型
from ultralytics import YOLO

# 读取视频
video_path = "E:\\test\\testVideo\\doro3.mp4"
cap = cv2.VideoCapture(video_path)

# 加载训练的模型
# model = YOLO('D:/Code/Python/testYolo11/src/runs/detect/train/weights/best.pt')
model = YOLO('D:/Code/Python/testYolo11/src/runs/train/exp/weights/best.pt')
# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 播放视频
while True:
    ret, frame = cap.read()
    if not ret:
        # 循环播放视频
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    # 模型推理
    results = model(frame)
    # 获取预测结果
    # 遍历检测结果并绘制
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < 0.5:  # 只显示置信度大于0.5的框
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[int(cls)]
        # 输出结果
        print(f"检测到：{class_name}, 置信度：{conf:.2f}")
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # 显示当前帧
    cv2.imshow("Video", frame)

    # 按下 'a' 键暂停
    if cv2.waitKey(1) & 0xFF == ord('a'):
        while True:
            # 等待用户按下 'r' 键继续
            if cv2.waitKey(1) & 0xFF == ord('d'):
                break
            # 显示当前帧
            cv2.imshow("Video", frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象和关闭所有窗口
cap.release()
cv2.destroyAllWindows()
