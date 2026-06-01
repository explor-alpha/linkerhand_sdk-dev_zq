# src/utils/llm_api.py
import json
import re
from openai import OpenAI
import time

class LLMDataProcessor:
    def __init__(self):
        # ==========================================
        # ⚠️ 请在这里填入你的 API 配置 ⚠️
        # 例如使用 DeepSeek (极其便宜且强大):
        # base_url="https://api.deepseek.com", api_key="sk-xxxx"
        # ==========================================
        self.client = OpenAI(
            api_key="sk-PJIehhIvOJAvkMGXvCfsd5L4BeFjgj7z54sfloPCAEVLvxof", 
            # gpt-5.2, gpt-5.1, gpt-5, gpt-4o，gpt-4.1一天5次；
            # 支持deepseek-r1, deepseek-v3, deepseek-v3-2-exp一天30次，
            # 支持gpt-4o-mini，gpt-3.5-turbo，gpt-4.1-mini，gpt-4.1-nano, gpt-5-mini，gpt-5-nano一天200次
            base_url="https://api.chatanywhere.tech"  # 国外使用：https://api.chatanywhere.org
        )
        self.model_name = "gpt-5-nano" # 替换为你使用的模型名，如 "deepseek-chat"

        # 定义分类目录列表
        self.valid_categories = [
            "Power_Cylinder", "Power_Sphere", "Precision_Pinch_2", 
            "Precision_Pinch_3", "Precision_Lateral", "Gesture_Open", 
            "Gesture_Fist", "Gesture_Social", "Single_Index", 
            "Single_Thumb", "Single_Middle", "Single_Ring", 
            "Single_Pinky", "Extent_Half", "Extent_Slight"
        ]

    def _call_llm(self, system_prompt, user_prompt):
        """底层调用大模型的封装方法，带有基本的重试机制"""
        for attempt in range(3): # 失败重试3次
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3 # 分类和生成结构化数据，温度放低一点
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️ API 调用出错: {e}，正在重试 ({attempt+1}/3)...")
                time.sleep(2)
        return None

    def assign_category(self, text):
        """
        函数 1: 根据 text 自动分配 category
        """
        system_prompt = f"""
你是一个机器人抓取分类学（Grasp Taxonomy）专家。
你的任务是将用户输入的自然语言控制指令，严格分类到以下列表中的【唯一一个】Category中：
{', '.join(self.valid_categories)}

**分类标准参考**：
1. Power Grasp(力量/包络)类型的 Category : Power_Cylinder(圆柱抓握：如握水杯、握锤子), Power_Sphere(球状抓握：如抓苹果、抓网球)
2. Precision Grasp(精准捏)类型的 Category : Precision_Pinch_2(二指捏：大拇指+食指，如捏针、捏硬币), Precision_Pinch_3(三指捏：大拇指+食指+中指，如拿笔、捏草莓), Precision_Lateral(侧捏 / 钥匙捏：大拇指压在食指侧面，用于插钥匙)
3. Gestures(手势)类型的 Category : Gesture_Open(全开：极限张开、放松), Gesture_Fist(握拳), Gesture_Social(社交手势：如 OK、点赞、比耶、竖中指)
4. Decoupled(单指控制)类型的 Category : Single_Index(食指), Single_Thumb(大拇指) ,Single_Middle(中指),Single_Pinky(无名指),Single_Ring(小指)
5. Continuous(连续程度)类型的 Category : Extent_Half(半开半合), Extent_Slight(轻微发力)

**要求**：
只输出分类标签（Category），绝对不要输出任何标点符号、解释或其他文字！
示例：
- “用力握紧那个水杯。” -> Power_Cylinder
- “把手掌包起来，抓稳这个球。” -> Power_Sphere
- “四根手指全部弯曲到底，大拇指扣在上面。” -> Power_Cylinder
- “用大拇指和食指轻轻捏住硬币。” -> Precision_Pinch_2
- “像拿笔一样，三根手指聚拢。” -> Precision_Pinch_3
- “大拇指侧着压在弯曲的食指上，准备开门。” -> Precision_Lateral
"""

        user_prompt = f"指令文本：{text}"
        
        result = self._call_llm(system_prompt, user_prompt)
        
        # 结果清洗：防止大模型擅自加句号或者输出不在列表里的词
        for cat in self.valid_categories:
            if cat.lower() in result.lower():
                return cat
        print(f"⚠️ 警告：大模型输出了未知的分类：{result}，默认归入 Gesture_Social")
        return "Gesture_Social"

    def generate_synonyms(self, text):
        """
        函数 2: 根据 text 和 category 进行 10 倍同义词数据增强
        返回一个包含 10 个字符串的 Python 列表
        """
        system_prompt = """
你现在是一个专业的机器人学习（Robot Learning）数据增强专家。
我正在训练一个神经网络，目的是将人类的自然语言指令映射为6自由度机械手的关节角度。
请根据用户提供的原始指令（{text}），帮我做同义词扩充的数据增强，生成10句意思相同，但表达方式和侧重点不同的指令。

请严格按照以下5个维度进行生成（每个维度生成2句，共10句）：
1. 目标导向型（Focus on goal）：不说具体手指，只强调要完成的任务或抓取的物体状态。
2. 动作分解型（Kinematic）：精确描述具体哪几根手指弯曲，哪几根伸直。
3. 日常口语型（Colloquial）：用普通人随口说出的指令，且可以带有口语化的模糊词或语气词。
4. 拟人/比喻型（Metaphorical）：用形象的比喻来描述这个手部姿态。
5. 复杂冗余型（Noisy）：句子里包含一些废话或环境描述，但核心指令不变。
注意，只输出纯净的指令，不要把prompt加进去

例如：
假设输入text是：“完全张开手掌”，  
LLM 将生成：
- "把手上的东西全都放开。" / "准备接收一个大物件，把手腾出来。"
- "大拇指侧展到最大，其余五根手指全部伸直至0度。" / "所有关节解除弯曲状态，保持在最大开合角。"
- "把手摊开吧。" / "手彻底张开。"
- "把手掌张得像一把平坦的扇子一样。" / "像投降一样把手全部展平。"
- "现在暂时不需要抓东西了，你先把手掌完全张开休息一下。" / "前面的桌子上没什么要拿的，让五根手指彻底伸直就行。"

⚠️【强制输出格式要求】⚠️：
你必须且只能输出一个合法的 JSON 字符串数组，不要包含任何 Markdown 标记 (如 ```json) 或额外的解释！
例如：
["把手上的东西全都放开。", "大拇指侧展到最大，其余五指伸直至0度。", "把手摊开吧。", "像投降一样把手全部展平。", "现在的桌子上没什么要拿的，你先把手掌张开休息一下。", "...(共10句)"]
"""
        user_prompt = f"原始指令：{text}\n，请生成10条增强数据（JSON格式）："
        
        result_str = self._call_llm(system_prompt, user_prompt)
        
        if not result_str:
            return []

        # 尝试解析大模型返回的 JSON
        try:
            # 清理可能的 markdown 标记
            result_str = re.sub(r'```json\s*', '', result_str)
            result_str = re.sub(r'```\s*', '', result_str)
            synonyms_list = json.loads(result_str)
            
            if isinstance(synonyms_list, list) and len(synonyms_list) > 0:
                return synonyms_list
            else:
                return []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 解析失败，大模型没有按格式输出。原始输出：\n{result_str}")
            return []