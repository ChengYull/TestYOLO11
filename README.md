# 基于 Ultralytics YOLO 的目标检测项目

## 项目简介
本项目基于 [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) 进行二次开发，实现了自定义数据集的目标检测模型训练与视频推理。

详细教程请参考 [博客园](https://www.cnblogs.com/1873cy/p/18844467)。

## 训练结果示例
50轮次结果：
![](https://img2024.cnblogs.com/blog/2734270/202505/2734270-20250506172016771-1610865420.gif)
100轮次：
![](https://img2024.cnblogs.com/blog/2734270/202505/2734270-20250506172024716-1809560716.gif)
## 环境依赖
- Anaconda 3.8+
- PyTorch 2.0+
- Python 3.8+
- pip
- 主要依赖见 `requirements.txt`

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集
> 数据集目录结构示例：
```bash
train/images/  # 存放图片
train/labels/  # 存放标注文件
train/doro.yaml  # 数据集配置文件
```
数据集配置文件 `doro.yaml` 示例：
```yaml
train: ../train/images/
val: ../train/images/
# number of classes
nc: 1

# class names
names: ['doro']
```

## 训练
执行训练脚本：
```python
# 导入警告模块并忽略警告信息
import warnings
warnings.filterwarnings('ignore')
# 导入YOLO模型
from ultralytics import YOLO

if __name__ == '__main__':
    # 创建YOLO模型实例，指定模型配置文件路径
    model = YOLO(model='D:/Code/Python/testYolo11/ultralytics/cfg/models/11/yolo11.yaml')
    
    # 开始训练模型
    model.train(
        data=r'D:/Code/Python/testYolo11/train/doro.yaml',  # 数据集配置文件路径
        imgsz=640,                    # 输入图像大小
        epochs=50,                    # 训练轮次数
        batch=4,                      # 批次大小
        workers=0,                    # 数据加载的工作进程数，0表示仅使用主进程
        device='0',                    # 训练设备，0表示使用第一个GPU，'cpu'表示使用CPU
        optimizer='SGD',              # 优化器类型，使用随机梯度下降
        close_mosaic=10,             # 在最后10个epoch关闭马赛克数据增强
        resume=False,                 # 是否从断点继续训练
        project='runs/train',         # 训练结果保存的项目目录
        name='exp',                   # 实验名称
        single_cls=False,             # 是否作为单类别检测
        cache=False,                  # 是否缓存图像到内存中以加快训练
    )
```

## 测试
执行测试脚本：
```python
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
```
