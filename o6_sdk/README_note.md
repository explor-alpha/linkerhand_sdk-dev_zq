> 建议： 台式机上配置一个用于训练; MacOS上配置一个用于展示 demo

## SDK 环境配置：

> 官方SDK 下载：[Linkerhand Python SDK](https://github.com/linker-bot/linkerhand-python-sdk)   
> **注意是：release_3.0.2（本项目灵巧手硬件是RS485，只能配置modbus；不能can！新版似乎有不兼容问题）**  

```bash
conda create -n linkerhand python=3.10 -y
conda activate linkerhand

cd o6_sdk
pip install -r requirements.txt
```


## 修改sdk-config：   

> 定位：LinkerHand/config/setting.yaml  
> MODBUS接口  

```zsh
ls /dev/tty.*
```


## 检验

- 运行：example/gui_control/gui_control.py  
- 运行：test_sdk.py     


## 项目管理：

> `o6_sdk`: 删除 `.git` `.gitattributes`  


