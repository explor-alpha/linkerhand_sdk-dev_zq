# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd

class HandMotionDataset(Dataset):
    def __init__(self, csv_file, text_extractor, device):
        print(f"📁 正在读取数据: {csv_file}")
        self.df = pd.read_csv(csv_file)
        
        # 1. 预提取文本 Embedding (耗时操作，仅执行一次)
        texts = self.df['text'].tolist()
        print("🧠 正在使用预训练模型提取文本特征 (可能需要几十秒)...")
        self.text_embeddings = text_extractor.encode(texts).to(device)
        
        # 2. 提取类别 ID
        self.category_ids = torch.tensor(self.df['category_id'].values, dtype=torch.long).to(device)
        
        # 3. 提取目标关节角度并归一化 (0~255 -> 0.0~1.0)
        target_cols = [
            'thumb_cmc_pitch', 'thumb_cmc_yaw', 'index_mcp_pitch', 
            'middle_mcp_pitch', 'pinky_mcp_pitch', 'ring_mcp_pitch'
        ]
        targets_raw = self.df[target_cols].values
        self.target_joints = torch.tensor(targets_raw, dtype=torch.float32).to(device) / 255.0
        
        # 4. 初始化当前状态 (这里先用全开状态占位，全部为 250/255)
        self.current_joints = torch.full_like(self.target_joints, 0.980392).to(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.text_embeddings[idx],
            self.category_ids[idx],
            self.current_joints[idx],
            self.target_joints[idx]
        )