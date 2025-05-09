# 转换模型为 TorchScript 格式 供C++调用
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('D:/Code/Python/testYolo11/src/runs/train/exp/weights/best.pt')

# 导出为 TorchScript 格式
model.export(format='torchscript')