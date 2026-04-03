from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


# —— 折扣判断 ——
class DiscountDecision(BaseModel):
    """折扣合规判断"""
    approved: bool = Field(description="是否批准折扣")
    max_discount_pct: float = Field(
        ge=0, le=100,
        description="最大可批准的折扣百分比")
    reason: str = Field(description="判断的依据，必须引用具体政策或数据")
    risk_level: Literal["low", "medium", "high", "unknown"] = Field(
        description="风险等级，基于毛利风险和政策偏离程度")
    policy_references: list[str] = Field(
        default=[],
        description="引用的政策条款列表，必须具体到条款编号或名称"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="对判断的置信度, 0-1 之间"
    )


# —— 政策违规检测 ——
class PolicyViolation(BaseModel):
    """政策违规检测"""
    order_id: str
    violation_type: Literal[
        "discount_exceeded",
        "margin_below_threshold",
        "unauthorized_bundle",
        "contract_term_mismatch"
    ]
    severity: Literal["warning", "critical"]
    description: str
    recommended_action: str


# —— 批量检查结果 ——
class BatchCheckResult(BaseModel):
    """批量订单检查结果"""
    total_orders: int
    violations: list[PolicyViolation]
    clean_orders: int
    summary: str

# —— 企业合同查询结果 ——
class ContractQueryResult(BaseModel):
    """供应商合同条款查询结果"""
    vendor_name: str
    payment_terms: Optional[str] = None
    delivery_sla_days: Optional[int] = None
    key_clauses: list[str] = Field(default=[])
    source_document: str
    page_reference: Optional[str] = None
    confidence: float = Field(ge = 0, le = 1)