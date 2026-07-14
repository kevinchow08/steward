"""Week 1:纯规则分类器——扩展名 + magic bytes,不涉及任何模型。"""


def load_rules(rules_path):
    # TODO: 读 config/rules.yaml,返回规则数据结构
    raise NotImplementedError


def classify_file(file_path, rules):
    # TODO: 先查 magic bytes,再查扩展名,都查不到返回 unknown
    # 返回结构待定,至少要包含 basic_type 和 matched_by(排查规则要不要扩充用)
    raise NotImplementedError
