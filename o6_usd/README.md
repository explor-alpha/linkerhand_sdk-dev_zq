# README：usd 资产准备

## 文件结构

> 注意文件结构不要轻易变动！usd 中有 references 关系！  

```
linkerhand_sdk-dev_zq/    
├── o6_urdf/             # 模型文件（Linkerhand O6 官方 URDF）
└── o6_usd/              # 模型文件（Isaac Sim 自定义 usd）   
    │
    ├── linkerhand_o6_right/         # 灵巧手 usd （独立资产）
    │   ├── configuration
    │   └── linkerhand_o6_right.usd  # Key: Isaac lab 中导入
    ├── my_asset                     # 独立资产
    │
    ├── scene1.usd                   # 场景建模，测试记录参数
    │
    ├── test_isaacsim/               # test-Isaac Sim (control & sensor)
    ├── test_isaaaclab/              # test-Isaac lab (control)
    │
    └── README.md
```


## Isaac 环境配置  

> 配置总览：
> 系统：本地；RTX 5060 Ti 16 GB；Linux；Ubuntu 22.04；  
> Isaac sim: v5.1.0；（安装方式：Pre-build binaries）  
> Isaac lab：v2.3.2；（安装方式：git/source code）  
> 项目级配置（conda）：`env_isaaclab`；python 3.11  

> 官方文档-链接 
> 1. Isaac sim documentation: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html  
> 2. Isaac lab documentation: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html  
> 3. Linkerhand O6 right 官方 URDF: https://github.com/linker-bot/linkerhand-urdf/tree/main/o6/right  

