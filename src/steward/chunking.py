"""把提取后的长文本切成适合 embedding 的片段。"""

import re
from dataclasses import asdict, dataclass

from steward.extractors import SHEET_MARKER_PREFIX, SHEET_MARKER_SUFFIX


@dataclass
class TextChunk:
    """一段可独立生成 embedding 的文本。"""

    index: int
    text: str
    start_offset: int
    end_offset: int

    def to_dict(self):
        return asdict(self)


# 找断点的优先级：越靠前越"结构完整"，找不到才退到下一档，硬切在字符位置
# 是最后的退路，不是默认行为——之前的版本只找换行符，一旦搜索范围内一个
# 换行符都没有（常见于没什么分段的长段落/长句子），就直接在字符位置硬切，
# 真实数据库里能看到大量单词被从中间切断的 chunk，就是这个原因。
# 每一档是 (要不要把这个分隔符本身吃掉, 分隔符列表)：
# - 换行/空格类分隔符本身不是内容，"吃掉"（断点在分隔符之前，分隔符归下一段，
#   会被下面的 lstrip 逻辑清掉）；
# - 句末标点是内容的一部分，断点要留在标点之后（标点归当前这段结尾），
#   不能把标点也切掉。
_BREAK_TIERS = [
    (True, ("\n\n", "\n")),
    (False, ("。", "！", "？", ".", "!", "?", "；", ";")),
    (True, (" ", "\t")),
]


def _find_break_point(text, search_start, search_end):
    """在 [search_start, search_end) 范围内，从后往前按 _BREAK_TIERS 的优先级
    找一个断点位置，用于确定"这一段在哪结束"。找到就返回；这一档、下一档都
    找不到，返回 None，交给调用方在 search_end 处硬切。
    """
    for strip_separator, separators in _BREAK_TIERS:
        for sep in separators:
            pos = text.rfind(sep, search_start, search_end)
            if pos > search_start:
                return pos if strip_separator else pos + len(sep)
    return None


def _find_break_point_forward(text, search_start, search_end):
    """跟 _find_break_point 逻辑一样，但从前往后找"从 search_start 开始，
    第一个断点在哪"，用于给下一段的重叠起点找一个干净边界。

    之前的版本没有这一步，重叠起点是简单地从上一段的结尾往回倒退固定字符数
    （`end - overlap_chars`），不管倒退到的位置是不是切在词/句子中间——中文
    没有空格，这个问题肉眼看不出来，但对英文这种有空格分词的文本，真实数据
    里能看到大量单词被从中间切断（比如"Clients"被切成"lients"）。找到断点后
    返回的位置不需要区分"要不要吃掉分隔符"——不管落在分隔符上还是标点后面，
    调用方那边已有的 lstrip 逻辑都会自动把开头多余的空白清掉。
    """
    for _, separators in _BREAK_TIERS:
        best = None
        for sep in separators:
            pos = text.find(sep, search_start, search_end)
            if pos != -1 and (best is None or pos < best):
                best = pos
        if best is not None:
            return best
    return None


def chunk_text(text, max_chars=800, overlap_chars=100):
    """按字符数切分文本，优先在段落/句子/单词边界断开，并保留少量重叠内容。

    overlap 的作用是避免一个语义句子刚好被切在两个片段之间，
    让相邻片段共享一小段上下文。
    """

    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars 必须大于等于 0 且小于 max_chars")

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        proposed_end = min(start + max_chars, len(text))
        end = proposed_end

        # 不在片段前半段断开，避免因为很早找到断点而产生过短片段。
        if proposed_end < len(text):
            break_point = _find_break_point(text, start + max_chars // 2, proposed_end)
            if break_point is not None:
                end = break_point

        raw_chunk = text[start:end]
        leading_spaces = len(raw_chunk) - len(raw_chunk.lstrip())
        trailing_spaces = len(raw_chunk) - len(raw_chunk.rstrip())
        actual_start = start + leading_spaces
        actual_end = end - trailing_spaces
        chunk_value = text[actual_start:actual_end]

        if chunk_value:
            chunks.append(
                TextChunk(
                    index=chunk_index,
                    text=chunk_value,
                    start_offset=actual_start,
                    end_offset=actual_end,
                )
            )
            chunk_index += 1

        if end >= len(text):
            break

        # 下一段回退一点，保留和上一段的上下文重叠——重叠起点也要对齐到干净的
        # 边界，不能简单倒退固定字符数（见 _find_break_point_forward 的注释）。
        next_start = end - overlap_chars
        if end < len(text):
            aligned = _find_break_point_forward(text, next_start, end)
            if aligned is not None and aligned > next_start:
                next_start = aligned
        start = max(next_start, start + 1)

    return chunks


# _extract_xlsx()/_extract_xls() 在多个工作表之间插入的分隔行，chunk_tabular_text()
# 靠它识别"一个新的表格块开始了"——不能把上一个工作表的表头带到下一个工作表的
# chunk 里。csv 没有这个标记，整份内容天然只有一个表格块。
#
# 正则从 extractors.py 的 SHEET_MARKER_PREFIX/SUFFIX 拼出来，不在这里重复写死
# 一份一样的中文字符串——标记格式改了，两边只需要改 extractors.py 那一处。
# re.escape() 是因为 "[" "]" 在正则里是特殊字符（字符集语法），拼进正则前
# 必须转义成字面量，否则会被解析成别的意思。
_SHEET_MARKER_PATTERN = re.compile(
    rf"^{re.escape(SHEET_MARKER_PREFIX)}.*{re.escape(SHEET_MARKER_SUFFIX)}$"
)


def _split_into_table_blocks(text):
    """把表格类提取文本按工作表边界拆成一个个"表格块"，每块是
    (表头行, 这块除表头外的数据行列表)。表头是这块里第一行有内容的行——
    对 csv 就是原本的列名行，对 xlsx 的某个工作表就是这个工作表自己的第一行。
    整块都是空行的情况直接跳过，不产出空表头的块。

    分两个阶段，不夹在一起做：
    第一阶段只管"分桶"——按标记行把所有行分进一个个桶（raw_blocks，
    每个桶是一份工作表的原始行），标记行本身不进任何桶，只负责触发"开一个
    新桶"。第二阶段才对每个桶做"拆表头/丢空桶"这件事，桶不管来自中间遇到
    标记行分出来的、还是最后一个没被后续标记行终结的，处理方式完全一样，
    不需要在扫描过程中特殊照顾"最后一块没人来收尾"这件事。
    """
    raw_blocks = [[]]
    for line in text.split("\n"):
        if _SHEET_MARKER_PATTERN.match(line.strip()):
            raw_blocks.append([])  # 开一个新桶，标记行自己不进桶
            continue
        raw_blocks[-1].append(line)

    blocks = []
    for lines in raw_blocks:
        first_content_idx = next((i for i, l in enumerate(lines) if l.strip()), None)
        if first_content_idx is None:
            continue  # 这个桶全是空行，没有表头也没有数据，跳过
        header = lines[first_content_idx]
        data_lines = lines[first_content_idx + 1:]
        blocks.append((header, data_lines))

    return blocks


def chunk_tabular_text(text, max_chars=800):
    """专门给表格类内容（csv/xlsx/xls 提取出来的、一行一条数据的文本）设计的
    切分函数，解决 chunk_text() 对表格数据的一个真实缺陷：普通切分只看字符
    数，切到文档中间的 chunk 天然看不到最前面的表头行，模型/搜索拿到这些
    chunk 时，是一堆脱离列名上下文的数字/文字，不知道对应哪一列。这里
    **每个 chunk 都带上它所属表格块的表头行**，不只是第一个 chunk。

    不像 chunk_text() 那样精确维护 start_offset/end_offset 对应原文里的
    精确字符位置——这两个字段现在全项目没有任何代码真正读取（只是存进
    数据库，没人用），这里退化成"这个 chunk 大致是从原文哪里开始的"，
    不值得为一个没人用的字段增加复杂度。

    不做重叠（不像 chunk_text 有 overlap_chars）——表格数据每一行本来就是
    独立的一条记录，行与行之间没有"一个句子被切断"这种需要靠重叠去弥补的
    语义连续性问题，表头本身已经把每个 chunk 需要的上下文补全了。
    """
    if not text or not text.strip():
        return []

    blocks = _split_into_table_blocks(text)
    chunks = []
    chunk_index = 0
    cursor = 0

    for header, data_lines in blocks:
        current_rows = []
        current_len = len(header)

        def _flush():
            nonlocal chunk_index, cursor
            if not current_rows:
                return
            body = "\n".join(current_rows)
            chunk_value = f"{header}\n{body}" if header else body
            chunks.append(TextChunk(
                index=chunk_index,
                text=chunk_value,
                start_offset=cursor,
                end_offset=cursor + len(body),
            ))
            chunk_index += 1
            cursor += len(body) + 1

        for row in data_lines:
            if not row.strip():
                continue
            if current_rows and current_len + len(row) + 1 > max_chars:
                _flush()
                current_rows = []
                current_len = len(header)
            current_rows.append(row)
            current_len += len(row) + 1
        _flush()

    return chunks
