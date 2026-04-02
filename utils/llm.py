# utils/llm.py

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat(
    user_message: str,
    system_prompt: str = "你是一个有帮助的助手。",
    model = "gpt-4o-mini",
    temperature: float = 0,
) -> str:

    resopnse = client.chat.completions.create(
        model = model,
        temperature = temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return resopnse.choices[0].message.content


if __name__ == "__main__":
    result = chat(
        user_message="客户问能不能给 15% 的折扣，我们公司最高折扣是 10%，我应该怎么回复？",
        system_prompt="你是一个 B2B 销售合规顾问，专门帮助销售团队判断报价是否符合公司政策。"
    )
    print(result)
    