# utils/llm.py

from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from typing import Type, TypeVar
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
T = TypeVar("T", bound=BaseModel)

def chat(
    user_message: str,
    system_prompt: str = "你是一个有帮助的助手。",
    model = "gpt-4o-mini",
    temperature: float = 0,
) -> str:
    """基础文字对话，返回string结果"""
    resopnse = client.chat.completions.create(
        model = model,
        temperature = temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return resopnse.choices[0].message.content

    
def structured_chat(
    user_message: str,
    response_schema: Type[T],
    system_prompt: str = "你是一个有帮助的助手, 所有回答以 JSON 格式输出。",
    model: str = "gpt-4o",
    temperature: float = 0,
) -> T:
    """
    结构化输出调用, 返回Pydantic为对象
    保证输出严格符合 response_schema 定义的格式
    """
    response = client.beta.chat.completions.parse(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        response_format=response_schema, 
    )
    return response.choices[0].message.parsed 