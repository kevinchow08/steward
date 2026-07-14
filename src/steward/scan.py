"""遍历目录,产出待分类的文件列表。"""

import os
from pathlib import Path


def iter_files(root_dir):
    # Path(...) 把字符串路径包装成 pathlib 的路径对象
    # .expanduser() 把 "~/Downloads" 这种带 ~ 的路径展开成真实的用户目录路径,比如 /Users/kevinchow/Downloads
    root = Path(root_dir).expanduser()

    # os.walk 会自顶向下遍历目录树,每进入一层目录就产出一个三元组:
    # dirpath  —— 当前所在的目录路径
    # dirnames —— 当前目录下的子目录名列表(还没进去)
    # filenames —— 当前目录下的文件名列表
    for dirpath, dirnames, filenames in os.walk(root):
        # 这是 os.walk 的一个常见用法:对 dirnames 做"原地修改"(用 [:] 赋值,而不是 dirnames = ...)
        # os.walk 在产出下一层之前会读取 dirnames 的当前内容来决定往哪些子目录递归,
        # 所以在这里把隐藏目录名过滤掉,os.walk 后面就不会再往 .git、.Trash 这些目录里钻了
        # 如果写成 dirnames = [...](重新绑定成一个新列表),os.walk 拿到的还是旧的引用,过滤就不生效
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            # 跳过隐藏文件,比如 .DS_Store
            # 注意:"." 前缀判断隐藏是 macOS/Unix 的命名约定,不是文件系统属性;
            # Windows 的隐藏文件靠元数据里的 Hidden 属性位,跟文件名无关,这里的判断搬到 Windows 上会失效。
            # 现阶段只跑 macOS,先不处理,真要兼容 Windows 时再加 os.stat().st_file_attributes 判断分支。
            if filename.startswith("."):
                continue
            yield Path(dirpath) / filename
