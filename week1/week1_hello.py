from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini", 
    temperature=0,
    messages=[
        {
            "role": "system",
            "content": "你是一个 B2B 销售合规顾问，专门帮助销售团队判断报价是否符合公司政策。"
        },
        {
            "role": "user",
            "content": "客户问能不能给 15% 的折扣，我们公司最高折扣是 10%，我应该怎么回复？"
        }
    ]
)

print(response.choices[0].message.content)
print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))