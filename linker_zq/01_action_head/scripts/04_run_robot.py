# scripts/03_run_robot.py
import torch
import os
import sys

# 保证能找到 config 和 src 目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.models.mlp_policy import DexHandMLP, TextFeatureExtractor
from src.utils.hardware_linker import LinkerHandController # 引入新的硬件控制器

def run_real_robot_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("⏳ 正在加载预训练语言大模型 (可能会耗时几秒)...")
    extractor = TextFeatureExtractor(config.TEXT_MODEL_NAME)
    
    print("⏳ 正在加载灵巧手 MLP 策略权重...")
    model = DexHandMLP().to(device)
    
    # 加载你通过 02_train.py 训练出的权重
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"找不到模型权重文件: {config.MODEL_SAVE_PATH}，请先运行 02_train.py！")
    
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval() # 开启推理模式
    
    # 🔗 初始化真机硬件 
    robot = LinkerHandController(speed=config.DEFAULT_SPEED, torque=config.DEFAULT_TORQUE)
    
    # 假设机械手初始状态是全开的 (全部为 250/255=0.980392)
    current_joints = torch.tensor([[0.980392, 0.980392, 0.980392, 0.980392, 0.980392, 0.980392]], dtype=torch.float32).to(device)
    robot.reset_to_open()
    
    print("\n" + "="*50)
    print("🎙️ NLP -> LinkerHand 真机控制终端已就绪！")
    print("输入 'q' 退出程序，输入 'r' 机械手复位。")
    print("="*50 + "\n")
    
    while True:
        text = input("\n🗣️ 请输入自然语言控制指令: ")
        if text.lower() == 'q':
            break
        if text.lower() == 'r':
            robot.reset_to_open()
            current_joints = torch.tensor([[0.980392, 0.980392, 0.980392, 0.980392, 0.980392, 0.980392]], dtype=torch.float32).to(device)
            continue

        cat_id = None
        if config.USE_CATEGORY:
            try:
                cat_id_str = input("👉 请输入动作类别 ID: ")
                cat_id = torch.tensor([int(cat_id_str)], dtype=torch.long).to(device)
            except ValueError:
                print("❌ 输入无效，请输入数字类别！")
                continue

        # 1. 文本转向量
        text_emb = extractor.encode([text]).to(device)
        
        # 2. 动态组装推理参数
        kwargs = {}
        if config.USE_CATEGORY:
            kwargs['cat_id'] = cat_id
        if config.USE_CURRENT_STATE:
            kwargs['curr_joints'] = current_joints
            
        # 3. AI 决策预测
        with torch.no_grad():
            target_normalized = model(text_emb, **kwargs)
           
        # 4. 反归一化为 0-255 指令并执行
        target_commands = (target_normalized.squeeze() * 255).round().int().tolist()
        robot.move_hand(target_commands)
        
        # 5. 更新状态记忆 (不管开没开当前状态开关，这句都留着，维护物理世界的真实状态)
        current_joints = target_normalized

    print("\n程序结束。为保障安全，机械手正在复位...")
    robot.reset_to_open()
    print("复位完成，再见！")

if __name__ == "__main__":
    run_real_robot_inference()