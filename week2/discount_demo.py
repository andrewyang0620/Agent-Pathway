"""
This is a final demo for week 2: 
Check if the product discount match the company policy
input: order info in natural language
output: structured decision output
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm import structured_chat, chat
from week2.schemas import DiscountDecision
from week2.system_prompt import COPILOT_SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

# defines the ploicy context that will be used in the system prompt
POLICY_CONTEXT = """
<policy_rules>
- 标准品类（硬件/软件）折扣上限：10%
- 服务类折扣上限：15%  
- 战略客户（年采购 > 100 万）特批上限：15%（需要销售总监签字）
- 折扣后毛利率不得低于：25%
- 单笔订单超过 50 万且折扣 > 8%：需要区域经理审批
</policy_rules>
"""
def analyze_discount_request(order_description: str) -> DiscountDecision:
    """分析折扣申请的合规性，返回结构化的判断结果""" 
    user_message =f"""
    {FEW_SHOT_EXAMPLES}
    {POLICY_CONTEXT}
    <current_request>
    {order_description}
    </current_request>
    请按以下步骤分析（Chain-of-Thought）：
    1. 确认客户类型和对应折扣上限
    2. 计算折扣后毛利率
    3. 检查是否需要额外审批
    4. 给出最终判断和最大允许折扣

    完成分析后输出结构化结论。
    """
    return structured_chat(
        user_message=user_message,
        response_schema=DiscountDecision,
        system_prompt=COPILOT_SYSTEM_PROMPT
    )

def display_decision(decision: DiscountDecision, order_desc: str):
    """用rich库美化输出决策结果"""
    risk_color = {"low": "green", "medium": "yellow", "high": "red", "unknown": "grey50"}
    status_text = "批准" if decision.approved else "拒绝"
    color = risk_color.get(decision.risk_level, "white")
    
    # 主结论
    console.print(Panel(
        f"[bold]{status_text}[/bold]\n"
        f"最大允许折扣：[cyan]{decision.max_discount_pct}%[/cyan]\n"
        f"风险等级：[{color}]{decision.risk_level.upper()}[/{color}]\n"
        f"置信度：{decision.confidence * 100:.0f}%",
        title="合规判断结果",
        border_style=color
    ))
    
    # 判断依据
    console.print(Panel(
        decision.reason,
        title="判断依据",
        border_style="yellow"
    ))
    
    # 政策引用
    if decision.policy_references:
        console.print("\n[bold]📎 引用来源：[/bold]")
        for ref in decision.policy_references:
            console.print(f"  • {ref}")


def main():
    # test cases
    test_cases = [
        {
            "name": "案例 1：普通客户超标折扣",
            "order": "客户：华东制造有限公司（年采购额 60 万，普通客户）\n产品：标准硬件设备\n请求折扣：12%\n当前毛利率：30%"
        },
        {
            "name": "案例 2：战略客户特批申请",
            "order": "客户：上海集成电路（年采购额 150 万，战略客户）\n产品：定制软件服务\n请求折扣：13%\n当前毛利率：35%"
        },
        {
            "name": "案例 3：毛利临界情况",
            "order": "客户：北京贸易公司（年采购额 80 万，普通客户）\n产品：标准软件许可\n请求折扣：8%\n当前毛利率：27%"
        },
    ]
    
    for case in test_cases:
        console.rule(f"[bold]{case['name']}[/bold]")
        console.print(f"\n[dim]订单信息：\n{case['order']}[/dim]\n")
        
        decision = analyze_discount_request(case["order"])
        display_decision(decision, case["order"])
        console.print("\n")

if __name__ == "__main__":
    main()