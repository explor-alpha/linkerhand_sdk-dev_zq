# Isaaclab 模板：Manager-based vs Direct：  
- Manager-based： 
    - 模块化（SceneManager 负责场景，ActionManager 负责动作映射，RewardManager 负责奖励函数等）
    - 动作/观察空间主要局限于简单的连续空间（'Box' 类型）
- Direct： 
    - 贴近底层（Gymnasium，step(), reset()）
    - 支持多智能体 (Multi-agent) 任务，并且支持更复杂的、非基础的动作/观察空间组合


# Isaaclab 算法库
- rl-games (~1X)
- rsl-rl (~1X)
- skrl (~1X)
    - 功能最全面。
    - 支持 PyTorch; JAX
    - 涵盖 AMP, IPPO, MAPPO 等，
    - 支持复杂空间（Fundamental/composite spaces）; 多智能体 Multi-agent
- sb3 (~0.03X)

# Isaaclab 项目文件结构 (TODO)

```
basic_grasp
 ┣ scripts
 ┃ ┣ skrl
 ┃ ┃ ┣ play.py
 ┃ ┃ ┗ train.py
 ┃ ┣ list_envs.py
 ┃ ┣ random_agent.py
 ┃ ┗ zero_agent.py
 ┣ source
 ┃ ┗ basic_grasp
 ┃ ┃ ┣ basic_grasp
 ┃ ┃ ┃ ┣ tasks
 ┃ ┃ ┃ ┃ ┣ manager_based
 ┃ ┃ ┃ ┃ ┃ ┣ basic_grasp
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ agents
 ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┗ skrl_ppo_cfg.yaml
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ mdp
 ┃ ┃ ┃ ┃ ┃ ┃ ┃ ┗ rewards.py
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ basic_grasp_env_cfg.py
 ┃ ┃ ┃ ┗ ui_extension_example.py
 ┣ README.md
 ┣ README_note.md
 ┣ isaac sim_linkerhand o6 场景建模.pdf

```

>  项目管理：删除内部项目 `.git` `.gitattributes`; 并将之合并至总项目的 `.git` `.gitattributes`

其中核心文件包含：
1. 训练 scripts/train.py: 用来启动强化学习训练的脚本。
2. 演示 scripts/play.py: 训练完成后，用来加载训练好的模型权重，观看机械臂实际抓取表现的推理脚本。
3. 环境 source/.../basic_grasp/
    - task_grasp_isaaclab_env_cfg.py: 环境的主配置文件。你的机械臂长什么样、桌子放在哪里、目标物体是什么、观察空间（State）包含哪些传感器数据、动作空间（Action）是控制关节位置还是力矩，全都是在这个文件里定义的。
    - mdp/rewards.py：定义mdp-step-reward; 写好的函数会被引入到上面的 env_cfg.py 中。
4. 参数 
    - source/.../basic_grasp/agents/skrl_ppo_cfg.yaml
    - source/.../basic_grasp/task_grasp_isaaclab_env_cfg.py

测试文件包含：
1. random_agent.py: 不使用agent，只发送随机动作指令。测试环境是否会崩溃、物理碰撞是否正常、看场景有没有穿模。
2. 

其他：
1. pyproject.toml / setup.py: Python 的包配置说明。它允许你通过 pip install -e . 的方式将你的抓取任务注册到你电脑的 Python 环境中，这样 scripts 里的代码才能 import 你的环境。(TODO: 虚拟环境直接用env_isaaclab?)
2. extension.toml: Isaac Sim/Lab 专用的扩展配置文件，声明了插件的名称、版本和依赖。
3. list_envs.py
4. zero_agent.py
5. ui_extension_example.py
(TODO)

笔记文件：
 ┣ README.md
 ┣ README_note.md
 ┣ isaac sim_linkerhand o6 场景建模.pdf


# mujoco+direct+SB3 ——> isaac+模块化+skrl
1. MuJoCo + SB3-SubprocVecEnv-cpu ——> isaac + skrl -gpu。速度 ; **并行训练数量**; **numpy to PyTorch Tensor**  (TODO)
2. isaac 调试方便

# Workflow
1. Isaac sim 创建 scene 并调试：
    1. 以物体为单位：
        1. 物理属性配置：在界面里直观地为物体添加碰撞体（Colliders）、刚体属性（Rigid Body）、质量（Mass）以及摩擦系数。
        2. 关节配置 (Articulation)：如果是自定义机械臂，在界面里设置好各个关节的运动范围（Limits）、阻尼（Damping）和驱动器（Drives）。
        3. 保存资产：将搭建好的物体或整个静态场景保存为一个或多个 .usd 文件（例如 my_custom_table.usd, weird_object.usd）。
        - PS：灵巧手（urdf导入 并固定手掌刚体并修改mimic联动关节定义）；双目深度相机（并设置忽略重力，调整好角度）; 要抓取的小球（并定义其摩擦，密度，大小）; 并定义好所有物体的位置角度
    2. 调整并初步定义物体之间几何关系（初始化）
2. Isaac Lab ：导入（prim_path; usd_path）; 向量化; 组装（初始化空间位置）

```这是一段概念代码，展示 Isaac Lab 如何读取 USD
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg

class MyGraspEnvCfg(ManagerBasedRLEnvCfg):

    target_object = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object", 
        spawn=sim_utils.UsdFileCfg(usd_path="path/to/your/weird_object.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1))
    )

    robot = ...
```

```
1. 需要你提供的代码文件或信息：

    task_grasp_isaaclab_env_cfg.py 的现有内容：这是环境的主配置文件，你可以把里面自动生成的代码发给我，我们直接在上面修改。
    mdp/rewards.py 的现有内容（如果有）：用来写奖励函数的地方。

2. 你在 Isaac Sim 中定义好的组件细节：

    灵巧手的 USD 路径，以及它的根链接（Root Link）名称和主动驱动关节（Active Joint Names）。
    深度相机的完整 Prim 路径（例如 /World/envs/env_0/Robot/palm_link/left_camera）。
    小球的 USD 路径或它的 Prim 名称。

3. 我们接下来的完善重点（工作路线）：
    重点攻克项目文件中以下四个核心模块
        资产加载与场景拼接（Scene Spawning）：在 EventCfg 或 SceneCfg 中正确配置灵巧手、小球和双目相机的路径，确保 Isaac Lab 启动时能正确在 GPU 中克隆出 4096 个独立的实验世界。

        动作空间映射（Action Management）：配置 ActionCfg，将神经网络输出的张量（Tensor）映射到你灵巧手的主动驱动关节上。由于你设置了 Mimic 联动关节，我们会确保动作维度只对应主动关节。

        观测空间构建（Observation Management）：
            状态部分：提取灵巧手的关节位置/速度、手掌相对于小球的相对位姿。
            视觉部分：配置相机的数据类型（distance_to_image_plane），将深度图像流转化为 PyTorch Tensor 喂给算法。

        奖励函数编写（Reward Functions in mdp/rewards.py）：为你量身定制一套适合灵巧手抓取的奖励机制（例如：手掌接近小球的距离惩罚、手指触碰小球的接触奖励、小球被抬高离开桌面的终极奖励）。
```


# skills & PS
1. Sim-to-Real & Domain Randomization
    - 加入 EventCfg (事件管理器)。虽然你在 USD 里设了摩擦力是 0.8，但你可以用代码让这 4096 个环境里的球，摩擦力在 0.5 到 1.0 之间随机浮动。
2. “深度相机”与 Observation 观察空间：
    - 必须配置 CNN（卷积神经网络）作为特征提取器 (Feature Extractor)