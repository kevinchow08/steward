"""Week 1:纯规则分类器——扩展名 + magic bytes,不涉及任何模型。"""

from pathlib import Path

import yaml

# 读文件开头多少个字节去做 magic bytes 比对
# 16 字节足够覆盖 rules.yaml 里目前最长的几条(都是 4 字节以内),留点余量方便以后加更长的规则
HEADER_READ_BYTES = 16


def load_rules(rules_path):
    # yaml.safe_load 把 yaml 文件解析成 Python 的嵌套 dict/list 结构
    # 结构跟 rules.yaml 里写的一样:{"document": {"extensions": [...], "magic_bytes": [...]}, ...}
    with open(rules_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_file(file_path, rules):
    file_path = Path(file_path)

    # 先看 magic bytes(读文件内容判断,更可靠)
    header_hex = _read_header_hex(file_path)
    category = _match_magic_bytes(header_hex, rules)

    if category == "archive":
        # zip 是个"容器"格式,docx/xlsx/pptx/odt 这些 Office/ODF 文档底层也是 zip,
        # 跟真正的 zip 压缩包开头字节完全一样,magic bytes 在这里不可信,要靠扩展名再确认一次
        ext_category = _match_extension(file_path, rules)
        if ext_category is not None and ext_category != "archive":
            return {"basic_type": ext_category, "matched_by": "extension_override"}
        return {"basic_type": "archive", "matched_by": "magic_bytes"}

    if category is not None:
        return {"basic_type": category, "matched_by": "magic_bytes"}

    # magic bytes 没匹配上,退回扩展名判断
    category = _match_extension(file_path, rules)
    if category is not None:
        return {"basic_type": category, "matched_by": "extension"}

    # 两边都没匹配上
    return {"basic_type": "unknown", "matched_by": None}


def _read_header_hex(file_path):
    # "rb" = 以二进制模式读(不按文本解码),避免遇到非文本文件(比如图片)报编码错误
    with open(file_path, "rb") as f:
        raw_bytes = f.read(HEADER_READ_BYTES)
    # bytes.hex() 把字节序列转成十六进制字符串,比如 b'\xff\xd8\xff' -> "ffd8ff"
    # .upper() 转大写,跟 rules.yaml 里写的大写十六进制对齐,避免大小写不一致导致匹配失败
    return raw_bytes.hex().upper()


def _match_magic_bytes(header_hex, rules):
    for category, spec in rules.items():
        for magic in spec.get("magic_bytes") or []:
            # startswith:只要文件开头字节能对上规则里的前缀,就算命中
            # (规则里的 magic bytes 通常比 HEADER_READ_BYTES 短,所以是"前缀匹配"而不是完全相等)
            if header_hex.startswith(magic.upper()):
                return category
    return None


def _match_extension(file_path, rules):
    ext = file_path.suffix.lower()  # .suffix 取文件名最后一个 "." 及之后的部分,比如 "a.PDF" -> ".PDF"
    for category, spec in rules.items():
        extensions = [e.lower() for e in spec.get("extensions") or []]
        if ext in extensions:
            return category
    return None
