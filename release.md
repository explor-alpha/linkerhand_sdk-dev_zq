### 版本号说明：

> 例如：v1.2.3（1/2/3/4）
> 1. "第一个1"对应"项目基座"(版本)
> 2. "第二个2"对应"上层模块"个数（仅最新版本），例如现在"上层模块"只有Policy_Network（Action_Head），则对应1
> 3. "第三个3"对应"算法骨架"个数（仅最新版本），例如现在只有MLP，则对应1

**版本更新原则:**
- 大版本，如 vx.x.x
	- "项目基座""上层模块"都是容纳"算法骨架"的容器，"算法骨架"是核心
	- 最新"项目基座""上层模块"版本: 需支持以往所有"算法骨架"(已弃用的除外)
	- 保存的"算法骨架"版本: 都是确认无误可以运行的版本
	- 保存的"算法骨架"版本: 有更新会在release中说明
- 小版本，如（1/2/3/4）
	- 只保存当前大版本的小版本
	- 用于记录：不稳定/正在更新的算法版本的backup

---
---
#### "项目基座"版本：

v0：官方SDK：Linkerhand_SDK

v1：基座搭建如附录

v2：基座搭建如附录
- TODO-优化布局：支持多"算法骨架"的对比分析
- TODO-优化布局：更加分布式，添加并剥离评估体系
- TODO-优化布局：更加分布式，添加并剥离data体系
- TODO-优化布局：更加分布式，添加并剥离loss function配置

---
#### "算法骨架"

1. **model_1_MLP**
```
text(自然语言)+(消融实验：current joints & Category) → embedding（text2vec） → Pytorch MLP → joint
```

- v1.1.1更新：
	
	v1.1.1(1):
	- 架构：搭建embedding（text2vec） → Pytorch MLP基本框架
	- 消融实验 Ablation Study: USE_CATEGORY参数
	- train-添加技巧：支持断点续训 (Resume Training)
	
	v1.1.1(2):
	- 消融实验 Ablation Study：USE_CURRENT_STATE参数
	- 消融实验 Ablation Study-添加技巧: `kwargs`动态组装
	- 数据分析：添加测试集&验证集
	- 数据分析：接入tensorboard
	- Key-debug：train逻辑问题： total_loss += loss.item() * len(batch_target)
	- debug：model：nn.Dropout(max(0, dropout - 0.1)),
	- debug：硬件初始化修复成250/255


2. **model_2_Cross-modal-Alignment**

	v2.1.2(1)



---
---
# 附："项目基座"

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
