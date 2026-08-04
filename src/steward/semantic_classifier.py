"""端侧文档语义分类与打标签引擎。

根据文档文本与元数据提取主分类、多标签、置信度及推理依据。
严格遵循低置信度（< 0.70）自动回退为 unclassified 的兜底原则。
"""

from dataclasses import dataclass
from typing import List


# 默认置信度回退阈值
CONFIDENCE_THRESHOLD = 0.70


@dataclass
class ClassificationResult:
    """分类与打标签的标准输出契约。"""

    category: str
    tags: List[str]
    confidence: float
    status: str
    reasoning: str


def classify_document_text(text: str, file_path: str = "", basic_type: str = "document") -> ClassificationResult:
    """分析文档正文，输出结构化的分类、标签与置信度。

    当前第一版使用确定性的规则/关键词提取引擎，
    后续可无缝替换或接入本地轻量 SLM 大模型推理。
    """

    if not text or not text.strip():
        return ClassificationResult(
            category="unclassified",
            tags=[],
            confidence=0.0,
            status="unclassified",
            reasoning="文档没有提取到有效文字内容",
        )

    text_lower = text.lower()
    path_lower = file_path.lower()

    # 1. 尝试匹配定义好的业务主分类规则
    matched_category = None
    confidence = 0.0
    reasoning = ""
    extracted_tags = []

    # 规则 1：招聘简历类
    if any(k in text_lower or k in path_lower for k in ("简历", "resume", "cv", "求职", "工作经历", "项目经验")):
        matched_category = "招聘简历"
        confidence = 0.95
        reasoning = "正文中包含简历、工作经历或项目经验等典型词汇"
        extracted_tags.extend(["简历", "人力资源"])

    # 规则 2：销售与财务数据类
    elif any(k in text_lower or k in path_lower for k in ("销售数据", "财报", "报销", "发票", "收支", "资产配置", "投资", "bogleheads")):
        matched_category = "财务销售"
        confidence = 0.90
        reasoning = "正文中包含财报、报销、发票或投资配置等财务特征"
        extracted_tags.extend(["财务", "数据"])

    # 规则 3：代码与开发工具/技术文档
    elif any(k in text_lower or k in path_lower for k in ("agent", "prompt", "mcp", "python", "javascript", "架构", "api", "git")):
        matched_category = "技术文档"
        confidence = 0.88
        reasoning = "正文中包含 Agent、API、架构或编程语言相关词汇"
        extracted_tags.extend(["技术", "开发"])

    # 规则 4：工作周报/总结类
    elif any(k in text_lower or k in path_lower for k in ("周报", "月报", "总结", "weekly report", "工作计划")):
        matched_category = "工作周报"
        confidence = 0.85
        reasoning = "正文中包含周报、月报或工作总结特征"
        extracted_tags.extend(["工作", "总结"])

    # 规则 5：个人证件与合同类
    elif any(k in text_lower or k in path_lower for k in ("身份证", "护照", "合同", "协议", "身份证号")):
        matched_category = "证件合同"
        confidence = 0.92
        reasoning = "正文中包含合同、协议或证件信息"
        extracted_tags.extend(["合同", "重要证件"])

    # 2. 提取泛化标签（根据高频词/环境词补充）
    if "agent" in text_lower:
        extracted_tags.append("Agent")
    if "2026" in text_lower or "2026" in path_lower:
        extracted_tags.append("2026")
    if "python" in text_lower:
        extracted_tags.append("Python")
    if "上海" in text_lower or "shanghai" in path_lower:
        extracted_tags.append("上海")

    # 去重并排序标签
    unique_tags = sorted(list(set(extracted_tags)))

    # 3. 校验置信度阈值：低于 0.70 触发 unclassified 兜底
    if matched_category is None or confidence < CONFIDENCE_THRESHOLD:
        return ClassificationResult(
            category="unclassified",
            tags=unique_tags,
            confidence=round(confidence, 2),
            status="unclassified",
            reasoning="内容无法高置信度归入已知主分类",
        )

    return ClassificationResult(
        category=matched_category,
        tags=unique_tags,
        confidence=round(confidence, 2),
        status="classified",
        reasoning=reasoning,
    )
