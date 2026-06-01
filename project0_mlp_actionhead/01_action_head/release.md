# release.md

> 采用标准的 语义化版本 (SemVer)：Major.Minor.Patch
- Major (大版本): 架构级调整（如：基座重构，不兼容旧代码）。
- Minor (小版本): 功能增加（如：新增了一个算法骨架，新增了一个评估体系）。
- Patch (补丁): 错误修复、参数微调。

## [v1.0.0] - 2026.04.15  

🚀 Added：  
- **新增 Project01 项目骨架：01_action_head**：  
	- utils：搭建`llm_api.py`：调用 LLM API 进行数据同义词增强以及数据分类  
	- utils：搭建`hardware_linker.py`：实现SDK串口通信、软防撞硬编码  
	- tensorboard  
- **新增算法 model01：Embedding+MLP**  
	- 基于 Pytorch 框架搭建 embedding+MLP 实现简单自然语言控制  
	- `text(自然语言)+(消融实验：current joints & Category) → embedding（text2vec） → Pytorch MLP → joint`  
	- 支持：消融实验 Ablation Study: USE_CATEGORY参数；USE_CURRENT_STATE参数；添加技巧: `kwargs`动态组装  
	- 支持：config.py：动态路径  
	- 支持：断点续训 (Resume Training)  

🛠 Fixed：  
- 统一所有硬件初始化操作：设置 250/255 状态为默认初始化状态  

⚙️ Changed：   
- debug in “model01：Embedding+MLP“：  
	- 添加测试集&验证集  
	- Key-debug：train 中 total_loss += loss.item() * len(batch_target)  
	- debug：model 中 nn.Dropout(max(0, dropout - 0.1))  

Project 文件结构：01_action_head 
```
        01_action_head/              <-- my project1主目录 (BASE_DIR)
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

TODO:  
- TODO-优化布局：支持多"算法骨架"的对比分析
- TODO-优化布局：更加分布式，添加并剥离评估体系
- TODO-优化布局：更加分布式，添加并剥离data体系
- TODO-优化布局：更加分布式，添加并剥离loss function配置
- **model_2_Cross-modal-Alignment**
