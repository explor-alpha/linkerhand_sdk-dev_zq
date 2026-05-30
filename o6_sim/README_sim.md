# Simulation: Linkerhand O6 right

## 🔥 News & Highlights  
  - **[TODO]** 测试isaacsim isaaclab; **RL训练简单抓取; 真机部署，测试sim2real**  
  - **[2026-05-30]**  本地配置isaacsim,isaaclab; isaacsim场景建模  

## 🎬 Show results

<table style="width: 100%; border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td width="33.3%" align="center" style="border: none;">
      <img src="show_results/show_scene1.gif" width="100%">
      <br><sub>Isaac Sim：场景建模</sub>
    </td>
    <td width="33.3%" align="center" style="border: none;">
      <img src="show_results/test_mimic.gif" width="100%">
      <br><sub>Isaac sim：Debug-mimic联动关节调试</sub>
    </td>
    <td width="33.3%" align="center" style="border: none;">
      </td>
  </tr>
</table>


## 文件结构

```
o6_sim/         
├── o6_urdf/                         # 模型文件（官方-urdf）
│   ├── meshes/                      
│   └── linkerhand_o6_right.urdf    
│ 
├── o6_usd/                          # 模型文件（Isaac Sim-usd）   
│   ├── myo6_zq.usd                  # Isaac Sim: 场景建模（导入linkerhand&debug-mimic联动关节; 添加物体&物理参数; 添加双目深度相机&视角; 设置root关节&固定; 等等）
│   ├── test_isaacsim/               # test-Isaac Sim (control & sensor)
│   └── test_isaaaclab/              # test-Isaac lab (control)
│
└── README_sim.md                    # Simulation
```

## 环境配置  

> 配置总览：
> 系统：本地；RTX 5060 Ti 16 GB；Linux；Ubuntu 22.04；  
> Isaac sim: v5.1.0；（安装方式：Pre-build binaries）  
> Isaac lab：v2.3.2；（安装方式：git/source code）  
> 项目级配置（conda）：`env_isaaclab`；python 3.11  

> 官方文档-链接 
> 1. Isaac sim documentation: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/quick-install.html  
> 2. Isaac lab documentation: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html  
> 3. Linkerhand O6 right 官方 URDF: https://github.com/linker-bot/linkerhand-urdf/tree/main/o6/right  

## 其他

> Linux mp4 转 gif  

```
cd projects_linux/own/linkerhand_sdk-dev_zq/show_results/
for f in *.mp4; do ffmpeg -i "$f" -vf "fps=18,scale=-1:600:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -vsync 1 -an "${f%.mp4}.gif"; done
```