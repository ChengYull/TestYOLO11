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
