"""Shared YAML frontmatter parser for memory and skills files."""

from dataclasses import dataclass, field


#FrontmatterResult为md文件的内容
#meta是---包裹中的内容，有name、description、type
#body是---之后的内容，是llm生成的这条记忆的原因及如何应用
@dataclass
class FrontmatterResult:
    meta: dict[str, str] = field(default_factory=dict)
    body: str = ""


def parse_frontmatter(content: str) -> FrontmatterResult:
    lines = content.split("\n")
    #第一行不是---，返回的实例的meta为空字典，body文件原文
    if not lines or lines[0].strip() != "---":
        return FrontmatterResult(body=content)
    #找闭合的---所在行数
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    #找不到闭合的---
    if end_idx == -1:
        return FrontmatterResult(body=content)

    meta: dict[str, str] = {}
    i = 1
    while i < end_idx:
        #找到":"所在索引
        colon_idx = lines[i].find(":")
        #没有":"则跳过
        if colon_idx == -1:
            i += 1
            continue
        #E：name: python风格。冒号前为key，后为value
        key = lines[i][:colon_idx].strip()
        value = lines[i][colon_idx + 1:].strip()
        # 如果冒号后面为"|", "|-", ">", ">-"，表示value的值跨行，说明当前在description，则value需要拼接跨行内容
        if value in ("|", "|-", ">", ">-"):
            value_lines = []
            i += 1
            while i < end_idx and lines[i].startswith("  "):
                value_lines.append(lines[i].strip())
                i += 1
            value = " ".join(value_lines)
        if key:
            meta[key] = value
        i += 1

    body = "\n".join(lines[end_idx + 1:]).strip()
    return FrontmatterResult(meta=meta, body=body)

