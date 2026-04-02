import json 
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def extract_json(text: str) -> dict:
    """直接提取json， 如果提取失败，尝试使用正则表达式提取"""
    
    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 ```JSON``` 块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
        
    # 尝试提取第一个 { ... } 块
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        
    raise ValueError("无法从文本中提取有效的 JSON 数据")


def openai_json_mode_sample():
    """专门针对 OpenAI 模型输出的 JSON 提取，优先提取 ```JSON``` 块"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format = {"type": "json_object"},  # 关键
        messages=[
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": "请给我一个 JSON 格式的回答，包含 name 和 age 两个字段。"}
        ]
    )
    
    result_text = json.loads(response.choices[0].message.content)
    return result_text

if __name__ == "__main__":
    openai_result = openai_json_mode_sample()
    print(openai_result)