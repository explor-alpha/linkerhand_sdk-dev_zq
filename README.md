#### Install SDK

> 官方SDK（linkerhand-python-sdk-main）下载：[linker-bot/linkerhand-python-sdk: Linkerhand Python SDK](https://github.com/linker-bot/linkerhand-python-sdk) 
> **注意是：release_3.0.2（本项目灵巧手硬件是RS485，只能配置modbus；不能can！新版似乎有不兼容问题）**

```bash
git clone ...
cd linkerhand-python-sdk/
conda create -n .. python=3.10
conda activate ...
pip install -r requirements.txt
```

- 修改config

## My project——QunZheng_Linker

#### 配置：

- Windows11
- SDK：release_3.0.2
- Python=3.10
- Windows + RTX 5060Ti (sm_120)
- CUDA 12.8 | torch 2.7.0

```bash
cd ...
conda activate ...
```

```
# Windows + RTX 5060Ti (sm_120)
# CUDA 12.8 | torch 2.7.0
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

```
# Windows + cpu for demo
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

```python
# 测试
import torch
print("PyTorch 版本:", torch.__version__)
print("内置 CUDA:", torch.version.cuda)
print("GPU 是否可用:", torch.cuda.is_available())
print("显卡名称:", torch.cuda.get_device_name(0))
print("计算能力:", torch.cuda.get_device_capability(0))
```

### v1.1.1 项目文件结构：

PS：本项目配置在windows系统：

```
cd C:\Users\QunZ\projects_win\own\linkerhand-python-sdk-main\
```

```
## v1(v1.1.1)
linkerhand-python-sdk-main/         <-- SDK_ROOT_DIR
├── LinkerHand/                     <-- 官方 SDK 包 (包含 api, utils 等)
├── ...
└── QunZheng_Linker/
    ├── QunZheng_backups/           <-- My关键版本备份
    │		├── release/                    # 版本说明
    │		└── ...
    │	
    └── Policy_Network/            <-- My当前项目主目录 (BASE_DIR)
		├── data/                       # 存放所有数据 (严禁将数据随代码一起传到Git)
		│   ├── backups/                # 1. 原始数据: 所有数据集
		│   ├── raw/initial_data.csv    # 2. 工作台；仅放入无增强数据进行数据增强
		│   └── processed/augmented_data.csv # 3. 工作台：仅放入增强后的数据
		│
		├── src/                        # 核心源码包 (面向对象设计)
		│   ├── __init__.py
		│   ├── models/                 # 网络架构定义
		│   │   ├── __init__.py
		│   │   └── mlp_policy.py       # PyTorch MLP及特征提取器代码
		│   ├── data/                   # 数据处理逻辑
		│   │   ├── __init__.py
		│   │   └── dataset.py          # PyTorch Dataset 定义
		│   ├── runs/                   # TensorBoard
		│   └── utils/                  # 工具箱
		│       ├── __init__.py
		│       ├── llm_api.py          # prompt定义、调用LLM API、数据增强的工具函数
		│       └── hardware_linker.py  # ~SDK串口通信、软防撞硬编码逻辑
		│
		├── scripts/                    # 执行脚本 Pipeline
		│   ├── 01_data_augment.py      # 步骤1：调用API进行数据增强—语义分组&同义词扩充
		│   ├── 02_train.py             # 步骤2：模型训练脚本
		│   └── 03_run_robot.py         # 步骤3：终端交互推理与真机控制
		│
		├── checkpoints/                # 模型权重 (*.pth)
		├── config.py                   # 全局配置文件 (动态路径、超参数)
		├── requirements.txt            # 依赖包列表
		└── README.md                   # 项目说明文档
```

### Workflow

**Step01. 数据增强&调整config.py**

> `01_data_augment.py`中定义了2种数据增强的函数。
> 1. 调用llm_api，自动补全`category`分类；
> 2. 调用llm_api，从5种维度进行同义词扩充。

**默认文件格式：**
- “自动补全category分类”函数的input形式是CSV；output形式是CSV
- “同义词扩充”函数的input形式是CSV；output形式是csv

**运行前检查：**
1. ==**raw\文件夹中仅有待处理数据：initial_data.CSV；且命名严格要求如此**==
2. ==**processed\中没有文件**==

**运行：**
```
python QunZheng_Linker\Policy_Network\scripts\01_data_augment.py
```

**运行后：**
- ==**移动文件至backups\**==
- ==**记录并手动保存输出的类别映射关系字典==**
- ==**根据python输出，修改`config.py`中的`NUM_CATEGORIES`参数！！！**==

![[data_aug_expand2.png]]


**Step02. train**

- **==修改`config.py`:**==
	- **==尤其是Category,current_joints相关参数==**
	- **==“训练动态超参数”**==

- 确认权重文件无误（没有 or 训练到一半）

- train
```bash
# 设置环境变量：huggingface镜像：
set HF_ENDPOINT=https://hf-mirror.com
python QunZheng_Linker\Policy_Network\scripts\02_train.py
```
- 支持断点续训，ctrl+c暂停。

Step03：TensorBoard & run

- 确认权重文件无误

- **==修改`config.py`:**==
	- **==尤其是Category,current_joints相关参数==**
	- **==“训练动态超参数”**==

- TensorBoard
```
tensorboard --logdir=QunZheng_Linker\Policy_Network\runs\
```

```
python QunZheng_Linker\Policy_Network\scripts\03_run_robot.py
```
