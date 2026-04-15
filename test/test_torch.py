# 测试
import torch
print("PyTorch 版本:", torch.__version__)
print("内置 CUDA:", torch.version.cuda)
print("GPU 是否可用:", torch.cuda.is_available())
print("显卡名称:", torch.cuda.get_device_name(0))
print("计算能力:", torch.cuda.get_device_capability(0))