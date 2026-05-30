# Real: Linkerhand O6 right

## 🔥 News & Highlights  
  - **[TODO]** 测试isaacsim isaaclab; **RL训练简单抓取; 真机部署，测试sim2real**  


## 🎬 Show results




## 文件结构

```
linkerhand_sdk-dev_zq/         
├── o6_real/  
│   └── README_real.md
└── sdk/                             #（SDK_ROOT_DIR）
    └──LinkerHand/                   # 官方 SDK 包 (包含 api, utils 等)
```

## 环境配置  


## 其他

> Linux mp4 转 gif  

```
cd projects_linux/own/linkerhand_sdk-dev_zq/show_results/
for f in *.mp4; do ffmpeg -i "$f" -vf "fps=18,scale=-1:600:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -vsync 1 -an "${f%.mp4}.gif"; done
```