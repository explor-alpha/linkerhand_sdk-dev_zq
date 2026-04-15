# src/models/mlp_policy.py
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import config  # 导入全局配置

class TextFeatureExtractor:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

class DexHandMLP(nn.Module):
    def __init__(self, text_dim=config.TEXT_DIM, curr_joint_dim=config.JOINT_DIM, hidden_dim=config.HIDDEN_DIM, dropout=config.DROPOUT_RATE):
        super().__init__()

        # 1. 基础维度：文本特征 (768)
        input_dim = config.TEXT_DIM
        
        # 2. 动态拼装维度：如果启用了 Category
        if config.USE_CATEGORY:
            self.cat_embedding = nn.Embedding(config.NUM_CATEGORIES, config.CAT_EMB_DIM)
            input_dim += config.CAT_EMB_DIM
            
        # 3. 动态拼装维度：如果启用了当前状态 (Current State)
        if config.USE_CURRENT_STATE:
            input_dim += config.JOINT_DIM

        # 4. 构建纯粹的 MLP 网络 (使用从 config 传入的 dropout)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(max(0, dropout - 0.1)), # 第二层稍微降低失活率

            nn.Linear(hidden_dim // 2, curr_joint_dim),
            nn.Sigmoid() 
        )

    # 给参数赋予默认值 None，方便外部动态调用
    def forward(self, text_emb, cat_id=None, curr_joints=None):
        features = [text_emb] # 列表用于存放所有要拼接的特征块
        
        if config.USE_CATEGORY and cat_id is not None:
            features.append(self.cat_embedding(cat_id))
            
        if config.USE_CURRENT_STATE and curr_joints is not None:
            features.append(curr_joints)
            
        # 一次性将所有激活的特征块在最后一个维度拼接起来
        x = torch.cat(features, dim=-1)
        
        return self.net(x)