# config.py
# 应用此文件，需要 import config！
import os
import sys

# ==========================================
# DIR & PYTHONPATH & DATA_PATH & TensorBoard
# ==========================================
# abspath(__file__)：config.py文件绝对路径
# os.path.dirname(): 当前路径的上一级文件夹Policy_Network的绝对路径
# linkerhand-python-sdk-main 根目录：config.py上推3级
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'o6_sdk')

# PYTHONPATH
# 1. Python 会默认把 当前运行文件所在目录 加到搜索路径
# 2. 将 SDK 根目录动态加入 Python 环境变量，使得 'import LinkerHand' 不会报错
if SDK_ROOT_DIR not in sys.path:
    sys.path.insert(0, SDK_ROOT_DIR)

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "initial_data.csv")
AUG_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "augmented_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "checkpoints", "best_policy.pth")
# 新增：TensorBoard 运行日志存储目录
TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "runs")

# ==========================================
# 🧠 模型结构超参数
# ==========================================
TEXT_MODEL_NAME = 'shibing624/text2vec-base-chinese'
if TEXT_MODEL_NAME == 'shibing624/text2vec-base-chinese':
    TEXT_DIM = 768      # 文本向量固定维度 # TODO-根据选用Embedding设置

HIDDEN_DIM = 256        # MLP 隐藏层宽度
DROPOUT_RATE = 0.3      # 神经元随机失活比例 (防过拟合) 
JOINT_DIM = 6           # 灵巧手自由度

# TODO: Experimental configurations are managed via a centralized config.py, with conditional logic to switch between different ablation studies.

# ==========================================
# 🌟 模型input超参数
# ==========================================
# 消融实验 Ablation Study：对比分析有无必要 input Category
USE_CATEGORY = True  
if USE_CATEGORY:
    NUM_CATEGORIES = 15     # 当前分类总数 # TODO-根据数据增强脚本的输出配置
    CAT_EMB_DIM = 16        # 分类特征映射维度 # TODO-根据数据增强脚本的输出配置

# 消融实验 Ablation Study：对比分析有无必要 input current joint
USE_CURRENT_STATE = False   

# ==========================================
# 🚀 训练动态超参数
# ==========================================
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-4    # 优化器学习率
WEIGHT_DECAY = 1e-4     # L2 正则化权重衰减 (防过拟合) 

# ==========================================
# LinkerHand 硬件配置
# ==========================================
HAND_TYPE = "right"     # 左右手
DEFAULT_SPEED = 50      # 默认运行速度
DEFAULT_TORQUE = 25     # 默认扭矩限制 (防撞断)