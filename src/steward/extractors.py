"""把支持的 document 文件统一提取成纯文本。

这一层只负责“读文件内容”，不负责 embedding、分类或索引。
不同格式的文件最后都会返回同一种 ExtractionResult，后面的分段和
向量化代码不需要知道原文件是 PDF 还是 DOCX。
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExtractionResult:
    """一次文本提取的统一结果。

    status 的可能值：
    - success: 成功提取到文本
    - no_text: 文件可读取，但没有提取到文字，例如扫描版 PDF
    - unsupported: 当前版本没有对应的提取器
    - error: 提取过程中发生异常
    """

    path: str
    status: str
    text: str = ""
    extractor: Optional[str] = None
    error: Optional[str] = None

    @property
    def char_count(self):
        """返回清洗后文本的字符数，方便报告统计。"""

        return len(self.text)

    def to_dict(self):
        """转成普通字典，后续可以直接写入 JSON 或 SQLite。"""

        result = asdict(self)
        result["char_count"] = self.char_count
        return result


# 这里列的是“当前提取器支持的后缀”，不等同于 Week 1 的全部 document 类型。
# 例如 .xls/.odt 仍然可以被 Week 1 识别为 document，但本阶段暂不支持其内容提取。
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
}


def extract_text(file_path):
    """提取一个文件的纯文本，并返回统一的 ExtractionResult。

    单个文件失败不会在这里向外抛出异常，而是返回 status="error" 和
    error 字段。这样批量处理时可以记录失败文件，继续处理其他文件。
    """

    path = Path(file_path).expanduser()
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        return ExtractionResult(
            path=str(path),
            status="unsupported",
            error=f"暂不支持的文件格式: {suffix or '(无扩展名)'}",
        )

    try:
        if suffix == ".pdf":
            text = _extract_pdf(path)
            extractor = "pypdf"
        elif suffix == ".docx":
            text = _extract_docx(path)
            extractor = "python-docx"
        elif suffix in {".html", ".htm"}:
            text = _extract_html(path)
            extractor = "beautifulsoup4"
        else:
            text = _extract_plain_text(path)
            extractor = "python-text"
    except Exception as exc:  # 单文件异常转为记录，不阻塞整批索引
        return ExtractionResult(
            path=str(path),
            status="error",
            extractor=extractor if "extractor" in locals() else None,
            error=f"{type(exc).__name__}: {exc}",
        )

    text = _normalize_text(text)
    status = "success" if text else "no_text"
    return ExtractionResult(
        path=str(path),
        status=status,
        text=text,
        extractor=extractor,
    )


def _extract_plain_text(path):
    """读取 TXT/MD，并兼容常见的 UTF-8、GB18030 编码。"""

    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "text",
        raw,
        0,
        len(raw),
        "无法按 UTF-8 或 GB18030 解码",
    )


def _extract_pdf(path):
    """提取 PDF 的文字层；扫描版 PDF 没有文字层时返回空字符串。"""

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _extract_docx(path):
    """提取 DOCX 段落和表格中的文字。"""

    from docx import Document

    document = Document(str(path))
    blocks = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            blocks.append(paragraph.text)

    # 表格不属于连续正文，但保留单元格之间的制表符，后续仍能检索到
    # 表格中的关键字；更复杂的表格结构化处理留到后续阶段。
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                blocks.append("\t".join(cells))

    return "\n".join(blocks)


def _extract_html(path):
    """去掉 HTML 中的脚本和样式，只保留可见文本。"""

    from bs4 import BeautifulSoup

    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    return soup.get_text("\n")


def _normalize_text(text):
    """做最小清洗，保留正文顺序，不做语义改写。"""

    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    # 连续空行压缩成一行，避免 HTML/PDF 的排版空白污染后续分段。
    normalized_lines = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = is_blank

    return "\n".join(normalized_lines).strip()
