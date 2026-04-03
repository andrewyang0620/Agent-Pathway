import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm import structured_chat
from week2.schemas import DiscountDecision
from week2.system_prompt import COPILOT_SYSTEM_PROMPT

# the prompt for the test case
ORDER_INFO = """
<order_info>
客户：科技股份有限公司（普通客户，年采购额 75 万元）
请求折扣：12%
产品品类：标准硬件
当前报价毛利率：29%
折扣后预计毛利率：22%
</order_info>

<policy>
- 标准品类折扣上限：10%
- 战略客户（年采购 > 100 万）可申请特批上限：15%
- 折扣后毛利率不得低于：25%
</policy>

请判断该折扣申请是否合规，并给出最大允许折扣。
"""

decision = structured_chat(
    user_message=ORDER_INFO,
    system_prompt=COPILOT_SYSTEM_PROMPT,
    response_schema=DiscountDecision
)

print(f"批准：{decision.approved}")
print(f"最大折扣：{decision.max_discount_pct}%")
print(f"风险等级：{decision.risk_level}")
print(f"置信度：{decision.confidence}")
print(f"理由：{decision.reason}")
print(f"政策引用：{decision.policy_references}")