from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from system_prompt import COPILOT_SYSTEM_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
class DiscountDecision(BaseModel):
    """带有类型约束的决策结果模型"""
    approved: bool
    max_discount_pct: float
    reason: str
    risk_level: Literal["low", "medium", "high", "unknown"]
    policy_reference: list[str]
    
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    temperature=0,
    messages=[
        {"role": "system", "content": COPILOT_SYSTEM_PROMPT},
        {"role": "user", "content": "..."}
    ],
    response_format=DiscountDecision
)

decision = response.choices[0].message.parsed
print(decision)
