# scripts/02_train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard
import os, sys
import math
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.models.mlp_policy import DexHandMLP, TextFeatureExtractor
from src.data.dataset import HandMotionDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")
    
    # ---------------------------------------------------------
    # 1. 初始化 TensorBoard Writer (架构师的魔法：动态命名)
    # ---------------------------------------------------------
    # 动态生成本次实验的名字，把消融实验的开关状态写在名字里！
    run_name = f"Cat_{config.USE_CATEGORY}_Curr_{config.USE_CURRENT_STATE}_{datetime.now().strftime('%m%d_%H%M')}"
    log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, run_name)
    writer = SummaryWriter(log_dir)
    print(f"📈 TensorBoard 日志将保存在: {log_dir}")

    # ---------------------------------------------------------
    # 2. 数据准备与三方划分 (Train / Val / Test)
    # ---------------------------------------------------------
    extractor = TextFeatureExtractor(config.TEXT_MODEL_NAME)
    full_dataset = HandMotionDataset(config.AUG_DATA_PATH, extractor, device)
    
    total_size = len(full_dataset)
    # 按照 70% 训练，15% 验证，15% 测试的比例划分
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    # 固定随机种子 42，保证每次运行划分的数据完全一样，实验才公平！
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"📊 数据集划分完毕: 训练 {train_size} | 验证 {val_size} | 测试 {test_size}")

    # ---------------------------------------------------------
    # 3. 初始化模型、优化器与损失函数
    # ---------------------------------------------------------
    model = DexHandMLP().to(device)

    # ==========================================
    # 🌟 架构师魔法：安全断点续训 (Safe Resume)
    # ==========================================
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"\n🔄 检测到历史模型权重 [{config.MODEL_SAVE_PATH}]")
        user_input = input("是否要接着上次的进度继续训练？(y/n) 默认 y: ")
        if user_input.lower() != 'n':
            try:
                # 尝试加载权重
                model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
                print("✅ 读档成功！将在现有大脑基础上继续进化...")
            except RuntimeError as e:
                # 核心防撞护栏：捕获消融实验导致的结构不匹配
                print("\n❌ 读档失败：检查config.py的消融实验开关 (如 USE_CATEGORY)！")
                print("🔧 已自动放弃读档，将为你从零开始训练这个全新的网络架构。\n")
        else:
            print("⚠️ 手动放弃读档，模型权重已重置，从零开始训练。")
    else:
        print("🌱 未检测到历史权重，从零开始训练。")
    # ==========================================

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')

    # ---------------------------------------------------------
    # 4. 开始训练大循环
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("🔥 开始训练网络...")
    for epoch in range(config.EPOCHS):
        # --- A. 训练阶段 (Train) ---
        model.train()
        total_train_loss = 0
        for batch_text, batch_cat, batch_curr, batch_target in train_loader:
            optimizer.zero_grad()
            
            # 动态组装消融实验参数
            kwargs = {}
            if config.USE_CATEGORY: kwargs['cat_id'] = batch_cat
            if config.USE_CURRENT_STATE: kwargs['curr_joints'] = batch_curr
                
            preds = model(batch_text, **kwargs)
            loss = criterion(preds, batch_target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_target)
            
        avg_train_loss = total_train_loss / train_size
        
        # --- B. 验证阶段 (Validation) ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_text, batch_cat, batch_curr, batch_target in val_loader:
                kwargs = {}
                if config.USE_CATEGORY: kwargs['cat_id'] = batch_cat
                if config.USE_CURRENT_STATE: kwargs['curr_joints'] = batch_curr
                    
                preds = model(batch_text, **kwargs)
                loss = criterion(preds, batch_target)
                total_val_loss += loss.item() * len(batch_target)
                
        avg_val_loss = total_val_loss / val_size
        
        # --- C. 记录 TensorBoard ---
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        
        # 打印日志
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:03d}/{config.EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
        # --- D. 早停与保存策略 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    print("✅ 训练完成！")
    
    # ---------------------------------------------------------
    # 5. 期末考试：在测试集 (Test Set) 上进行最终盲测
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("🎓 开始最终测试集盲测...")
    
    # 必须加载刚刚保存的“最好”的一代模型，而不是最后一代
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    
    total_test_loss = 0
    with torch.no_grad():
        for batch_text, batch_cat, batch_curr, batch_target in test_loader:
            kwargs = {}
            if config.USE_CATEGORY: kwargs['cat_id'] = batch_cat
            if config.USE_CURRENT_STATE: kwargs['curr_joints'] = batch_curr
                
            preds = model(batch_text, **kwargs)
            loss = criterion(preds, batch_target)
            total_test_loss += loss.item() * len(batch_target)
            
    avg_test_loss = total_test_loss / test_size
    writer.add_scalar('Loss/Test_Final', avg_test_loss, 0) # 记录最终成绩
    
    # 把数学 Loss 翻译成机器人的“物理偏差角度” (0-255制)
    physical_error = math.sqrt(avg_test_loss) * 255
    
    print(f"🏆 最终测试集 MSE Loss: {avg_test_loss:.6f}")
    print(f"📐 换算为物理关节指令 (0-255) 的平均绝对误差约为: ±{physical_error:.1f}")
    
    # 关闭 writer
    writer.close()
    print("="*40 + "\n")

if __name__ == "__main__":
    train()