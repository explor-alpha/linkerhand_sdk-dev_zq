# scripts/01_data_augment.py
import pandas as pd
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils.llm_api import LLMDataProcessor

def run_augmentation_category():
    print(f"📁 读取原始数据: {config.RAW_DATA_PATH}")
    try:
        df_raw = pd.read_csv(config.RAW_DATA_PATH,encoding='gbk')
    except FileNotFoundError:
        print(f"❌ 找不到文件 {config.RAW_DATA_PATH}！请检查路径。")
        return

    # 初始化大模型处理器
    llm = LLMDataProcessor()
    
    # ==========================================
    # 阶段 1：利用大模型填补缺失的 Category
    # ==========================================
    print("\n" + "="*40)
    print("🤖 阶段 1: 正在自动推断缺失的 Category ...")
    
    for index, row in df_raw.iterrows():
        # 如果 category 列是空值 (NaN) 或者纯空格
        if pd.isna(row['category']) or str(row['category']).strip() == "":
            text = row['text']
            print(f"分析文本 [{text}] -> ", end="", flush=True)
            
            predicted_cat = llm.assign_category(text)
            df_raw.at[index, 'category'] = predicted_cat # 更新表格
            
            print(f"[{predicted_cat}]")
            time.sleep(0.5) # 防止触发 API 频率限制

    os.makedirs(os.path.dirname(config.AUG_DATA_PATH), exist_ok=True)
    df_raw.to_csv(config.AUG_DATA_PATH, index=False,encoding='gbk')
    print("\n🎉 大功告成！")
    print(f"已保存至: {config.AUG_DATA_PATH}")


def run_augmentation_append():
    print(f"📁 读取原始数据: {config.RAW_DATA_PATH}")
    try:
        df_raw = pd.read_csv(config.RAW_DATA_PATH,encoding='gbk')
    except FileNotFoundError:
        print(f"❌ 找不到文件 {config.RAW_DATA_PATH}！请检查路径。")
        return

    # 初始化大模型处理器
    llm = LLMDataProcessor()

    # ==========================================
    # 阶段 2：利用大模型进行 10倍 数据增强
    # ==========================================
    print("\n" + "="*40)
    print("🚀 阶段 2: 正在根据 5 大维度生成增强数据 ...")
    
    augmented_rows = []

    for index, row in df_raw.iterrows():
        original_text = row['text']
        
        # 1. 无论如何，先把原始的这一行保存下来
        augmented_rows.append(row.to_dict())
        
        # 2. 调用大模型生成同义词
        print(f"⏳ 正在扩充: {original_text}  ...")
        synonyms = llm.generate_synonyms(original_text)
        
        if synonyms:
            print(f"   ✅ 成功生成 {len(synonyms)} 条增强数据")
            for syn_text in synonyms:
                # 复制原行的数据（包括6个关节角度和分类），只替换 text
                new_row = row.copy().to_dict()
                new_row['text'] = syn_text
                augmented_rows.append(new_row)
        else:
            print("   ❌ 生成失败，跳过该条数据。")
            
        time.sleep(1) # 保护 API

    # 生成增强后的 DataFrame
    df_aug = pd.DataFrame(augmented_rows)

    # ==========================================
    # 阶段 3：架构师的魔法 -> 将文本分类转换为神经网络需要的数字 ID
    # ==========================================
    print("\n" + "="*40)
    print("🔢 阶段 3: 映射 Category ID (适配神经网络) ...")
    
    # 提取所有出现过的独立类别
    unique_categories = df_aug['category'].unique().tolist()
    # 创建一个字典映射: {'Power_Cylinder': 0, 'Gesture_Open': 1, ...}
    cat_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
    
    # 打印映射关系表供用户参考
    print("类别映射关系字典:")
    for cat, idx in cat_to_id.items():
        print(f"  ID: {idx} <--> {cat}")
        
    # 在表格中新增一列 'category_id'
    df_aug['category_id'] = df_aug['category'].map(cat_to_id)

    # 自动更新 config.py 中的 NUM_CATEGORIES （非常重要！）
    print(f"💡 提示：请确保你 config.py 中的 NUM_CATEGORIES 被设置为 {len(unique_categories)}")

    # ==========================================
    # 阶段 4：保存最终文件
    # ==========================================
    os.makedirs(os.path.dirname(config.AUG_DATA_PATH), exist_ok=True)
    # 把 category_id 提到第二列方便查看
    cols = ['text', 'category', 'category_id'] + [c for c in df_aug.columns if c not in ['text', 'category', 'category_id']]
    df_aug = df_aug[cols]
    
    df_aug.to_csv(config.AUG_DATA_PATH, index=False, encoding='utf-8-sig')
    print("\n🎉 大功告成！")
    print(f"原始数据 {len(df_raw)} 条，增强后变为 {len(df_aug)} 条。")
    print(f"已保存至: {config.AUG_DATA_PATH}")

if __name__ == "__main__":
    #run_augmentation_category()
    run_augmentation_append()
    # 先运行 run_augmentation_category() 来填补缺失的分类标签，再运行 run_augmentation_append() 来生成增强数据并映射类别 ID。
    # 在运行 run_augmentation_category() 后需要手动移动生成的 augmented_data.csv 到 initial_data.csv 的位置，覆盖原始数据，以便 run_augmentation_append() 能够在完整的分类标签基础上进行增强。