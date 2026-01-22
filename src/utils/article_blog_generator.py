# ============================================================
# src/utils/article_blog_generator.py
# Tree-structured "decompose then backtrack merge" blog generator
# JSON FORMAT: ALWAYS [{"point": "..."}]
# IMPORTANT: EVERY STEP INCLUDES FULL ARTICLE CONTEXT
# ============================================================

import json
import re
from typing import List, Optional, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field

from src.services.llm import LLM
from src.services.custom_tools import Tool_Calls


# ============================================================
# Data models (Python-side tree ONLY)
# ============================================================

class NodeL2(BaseModel):
    point: str
    detail_markdown: Optional[str] = None


class NodeL1(BaseModel):
    point: str
    children: List[NodeL2] = Field(default_factory=list)
    section_markdown: Optional[str] = None


class OutlineTree(BaseModel):
    title: str
    children: List[NodeL1]


class BlogGenConfig(BaseModel):
    model_name: str
    api_key: str
    llm_url: Optional[str] = None

    l1_points: int = 5
    l2_points: int = 4

    language: Literal["zh", "en"] = "zh"
    style: Literal["math", "normal", "rigorous"] = "math"

    max_article_chars: int = 120000
    max_leaf_chars: int = 6000

    max_workers_step2: int = 4
    max_workers_step3: int = 8
    max_workers_step4: int = 4


# ============================================================
# Helpers
# ============================================================

def _extract_json_array(text: str) -> List[dict]:
    """
    Robust JSON array extraction:
    - find first '['
    - find last ']'
    - parse only inside
    """
    if not text:
        raise ValueError("Empty LLM output")

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in LLM output")

    sliced = text[start : end + 1]
    return json.loads(sliced)


def _json_array_rule() -> str:
    return (
        "你必须 **只输出一个合法 JSON 数组**，格式如下：\n\n"
        "[\n"
        '  { "point": "..." },\n'
        "  ...\n"
        "]\n\n"
        "❗ 只能是数组，不能是对象 `{}`\n"
        "❗ 不要输出解释、Markdown、代码块或任何多余文字\n\n"
        "❗ 必须式中文\n\n"
    )


def _style_rules(style: str) -> str:
    if style == "math":
        return "偏数学化：定义符号、强调假设、给出逻辑链。"
    if style == "rigorous":
        return "措辞严格，区分假设、证据、结论和局限。"
    return "正常技术博客风格，清晰直接。"


def _extract_fig_placeholders(md: str) -> List[str]:
    if not md:
        return []
    return re.findall(r"\[\[FIG:([0-9]+[a-z]?)\]\]", md)


def _new_llm(cfg: BlogGenConfig) -> LLM:
    return LLM(
        api_key=cfg.api_key,
        llm_url=cfg.llm_url,
        model_name=cfg.model_name,
        format="openai",
        ec=None,
    )


def _build_full_context(title, text, images, max_chars) -> str:
    return f"""
文章标题：
{title}

可用图片（只能使用 [[FIG:x]] 占位）：
{json.dumps(images, ensure_ascii=False)}

文章正文（全文上下文，每一步都提供，可能被截断）：
{text[:max_chars]}
"""


# ============================================================
# Core pipeline
# ============================================================

def generate_blog_from_article_tree(
    *,
    task_id: str,
    article_id: str,
    article_title: str,
    article_text: str,
    image_catalog: List[dict],
    config: BlogGenConfig,
    tc: Optional[Tool_Calls] = None,
) -> dict:

    full_context = _build_full_context(
        article_title,
        article_text,
        image_catalog,
        config.max_article_chars,
    )

    llm_main = _new_llm(config)
    # with open("/home/ubuntu/workspace/test.log", "a", encoding="utf-8") as f:
    #     print("start", file=f)

    # ========================================================
    # Step 1: Article → L1 points
    # ========================================================
    prompt_l1 = (
        full_context
        + f"""
任务：
- 从全文中提取 {config.l1_points} 个左右的「一级要点」
- 每个 point 是一句完整、可作为章节标题的陈述
- 可以从背景, 文章得出的结论等信息出发
- 不要编号，不要解释

风格：{_style_rules(config.style)}
"""
        + _json_array_rule()
    )

    raw_l1 = llm_main.query(prompt_l1, False)
    # with open("/home/ubuntu/workspace/test.log", "a", encoding="utf-8") as f:
    #     print("Article->L1:"+raw_l1, file=f)
    l1_items = _extract_json_array(raw_l1)

    tree = OutlineTree(
        title=article_title,
        children=[NodeL1(point=x["point"]) for x in l1_items if "point" in x]
    )

    # ========================================================
    # Step 2: L1 → L2 points（parallel）
    # ========================================================
    def expand_l1(idx: int, n1: NodeL1) -> Tuple[int, List[NodeL2]]:
        llm = _new_llm(config)
        prompt = (
            full_context
            + f"""
当前一级要点：
{n1.point}

任务：
- 为该要点生成 {config.l2_points} 个「二级子点」
- 每个 point 应当是可独立展开的论点
- 不要写解释

风格：{_style_rules(config.style)}
"""
            + _json_array_rule()
        )
        raw = llm.query(prompt, False)
        # with open("/home/ubuntu/workspace/test.log", "a", encoding="utf-8") as f:
        #     print("L1->L2:"+raw, file=f)
        arr = _extract_json_array(raw)
        return idx, [NodeL2(point=x["point"]) for x in arr if "point" in x]

    with ThreadPoolExecutor(max_workers=min(config.max_workers_step2, len(tree.children))) as pool:
        futures = [pool.submit(expand_l1, i, n1) for i, n1 in enumerate(tree.children)]
        for fut in as_completed(futures):
            i, children = fut.result()
            tree.children[i].children = children

    # ========================================================
    # Step 3: L2 → detail markdown（parallel）
    # ========================================================
    figs = [x.get("fig") for x in image_catalog if x.get("fig")]

    def write_leaf(i, j, n1, n2):
        llm = _new_llm(config)
        prompt = f"""
全文上下文：
{full_context}

你将围绕以下「二级子点」写一段详细解释（Markdown）。

一级要点：
{n1.point}

二级子点：
{n2.point}

规则：
- 可使用图片：{figs}
- 图片必须用 [[FIG:x]] 占位, 注意, 任何图片类似于[[FIG:1B]]这种是不能接受的, 必须写成[[FIG:1]], 然后你在引用的时候, 说明是B图
- 定义关键概念，逻辑自洽
- 使用中文
- 建议 400–900 字，不超过 {config.max_leaf_chars}

请开始：
"""
        md = llm.query(prompt, False)
        # with open("/home/ubuntu/workspace/test.log", "a", encoding="utf-8") as f:
        #     print("L2->detail:"+md, file=f)
        return i, j, md

    tasks = []
    for i, n1 in enumerate(tree.children):
        for j, n2 in enumerate(n1.children):
            tasks.append((i, j, n1, n2))

    with ThreadPoolExecutor(max_workers=min(config.max_workers_step3, len(tasks))) as pool:
        futures = [pool.submit(write_leaf, *t) for t in tasks]
        for fut in as_completed(futures):
            i, j, md = fut.result()
            tree.children[i].children[j].detail_markdown = md

    # ========================================================
    # Step 4: Merge L2 → L1 section（parallel）
    # ========================================================
    def merge_section(i, n1):
        llm = _new_llm(config)
        pack = [{"point": c.point, "detail": c.detail_markdown} for c in n1.children]
        prompt = f"""
全文上下文：
{full_context}

你将把多个子点解释整合为一个章节（Markdown）。

章节标题：
## {n1.point}

子点材料：
{json.dumps(pack, ensure_ascii=False)}

规则：
- 合并重复内容
- 可使用图片：{figs}
- 使用中文
- 文章中的需要用图解释的地方, 图片必须用 [[FIG:x]] 占位, 注意, 任何图片类似于[[FIG:1B]]这种是不能接受的, 必须写成[[FIG:1]], 然后你在引用的时候, 说明是B图
"""
        llm_raw = llm.query(prompt, False)
        # with open("/home/ubuntu/workspace/test.log", "a", encoding="utf-8") as f:
        #     print("detail->L2:"+llm_raw, file=f)
        return i, llm_raw

    with ThreadPoolExecutor(max_workers=min(config.max_workers_step4, len(tree.children))) as pool:
        futures = [pool.submit(merge_section, i, n1) for i, n1 in enumerate(tree.children)]
        for fut in as_completed(futures):
            i, sec = fut.result()
            tree.children[i].section_markdown = sec

    # ========================================================
    # Step 5: Final merge
    # ========================================================
    sections = "\n\n---\n\n".join(n.section_markdown or "" for n in tree.children)

    prompt_final = f"""
全文上下文：
{full_context}

章节内容：
{sections}

请将以下章节整合为一篇完整博客(Markdown)

要求：
- 标题：# {article_title}
- 开头 TL;DR (5-8 条)
- 可使用图片：{figs}
- 文章中的需要用图解释的地方, 必须使用 [[FIG:x]] 占位, 注意, 任何图片类似于[[FIG:1B]]这种是不能接受的, 必须写成[[FIG:1]], 然后你在引用的时候, 说明是B图
- 使用中文
- 讲清楚文章的背景内容, 得到的结论等关键信息
- 合并重复内容, 你最后需要给出的是一个讲解清楚的博客, 需要有你自己的逻辑链条
- 最后给 Figure Index 部分, 每张图必须引用 [[FIG:x]] (这样我才能看得见), 每张图请说明各个子图都是什么意思, 比如A, B, C, ...
- 任何数学公式, **不要使用()或者[], 正确的使用方法是$$, 一个例子是不要(\A_i\), (A_i), 而是$A_i$**
"""
    blog_md = llm_main.query(prompt_final, False)
    # with open("/home/ubuntu/workspace/test.log", "a", encoding="utf-8") as f:
    #     print("final:"+blog_md, file=f)

    return {
        "blog_markdown": blog_md,
        "used_figs": sorted(set(_extract_fig_placeholders(blog_md))),
        "tree": tree.model_dump(),
    }
