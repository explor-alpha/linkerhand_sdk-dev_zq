# Isaac 配置

## 文件结构

```
linkerhand_sdk-dev_zq/    
├── o6_urdf/             # 模型文件（Linkerhand O6 官方 URDF）
└── o6_usd/              # 模型文件（Isaac Sim 自定义 usd）   
    ├── myo6_zq.usd                  # Isaac Sim: 场景建模（导入linkerhand&debug-mimic联动关节; 添加物体&物理参数; 添加双目深度相机&视角; 设置root关节&固定; 等等）
    ├── test_isaacsim/               # test-Isaac Sim (control & sensor)
    └── test_isaaaclab/              # test-Isaac lab (control)
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

