"""把提取后的长文本切成适合 embedding 的片段。"""

from dataclasses import asdict, dataclass


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
