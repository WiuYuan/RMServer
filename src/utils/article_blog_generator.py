# src/utils/article_blog_generator.py
# ============================================================
# src/utils/article_blog_generator.py
# Tree-structured "decompose then backtrack merge" blog generator
# JSON FORMAT: ALWAYS [{"point": "..."}]
# IMPORTANT: EVERY STEP INCLUDES FULL ARTICLE CONTEXT
# ============================================================

import json
import logging
import re
from typing import List, Optional, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

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

    max_workers_step2: int = 3
    max_workers_step3: int = 3
    max_workers_step4: int = 3


# ============================================================
# Helpers
# ============================================================

# DOC-BEGIN id=extract_json_array#1 type=function v=2
# summary: 从LLM原始输出文本中提取JSON数组。依次尝试：(1)直接解析，(2)修复反斜杠后解析，(3)修复未转义双引号后解析，(4)正则逐条提取point字段作为最终兜底。返回List[dict]。
# intent: LLM输出经常包含非法JSON（未转义的反斜杠、嵌入的双引号、多余文本等），单一修复策略不够健壮。采用多级降级策略确保尽可能解析成功，只有完全无法提取时才抛异常。最后的正则兜底可能丢失point以外的字段，但对当前管线足够。
def _extract_json_array(text: str) -> List[dict]:
    """
    Robust JSON array extraction with multi-level fallback.
    """
    if not text:
        raise ValueError("Empty LLM output")

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found in LLM output: {text[:200]}")

    sliced = text[start : end + 1]

    # --- Level 1: 直接解析 ---
    try:
        return json.loads(sliced)
    except json.JSONDecodeError:
        pass

    # --- Level 2: 修复非法反斜杠 ---
    # DOC-BEGIN id=extract_json_array/fix_backslash#1 type=behavior v=1
    # summary: 用正则将不属于JSON标准转义序列的单反斜杠替换为双反斜杠，然后尝试解析
    # intent: LLM常输出LaTeX公式如 \alpha、\beta，这些在JSON字符串中是非法转义；但需保留合法转义如 \n \t \" 等
    fixed = re.sub(r'\\(?![u"bfnrt/\\])', r'\\\\', sliced)
    # DOC-END id=extract_json_array/fix_backslash#1
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # --- Level 3: 修复 point 值内部的未转义双引号 ---
    # DOC-BEGIN id=extract_json_array/fix_inner_quotes#1 type=behavior v=2
    # summary: 逐字符扫描JSON字符串，识别"point"字段值内部的未转义双引号并替换为中文引号，再尝试解析
    # intent: LLM经常输出 "point": "xxx "yyy" zzz" 这类嵌套双引号。通过状态机定位字段值的起止位置，
    #         将内部多余的双引号替换为中文引号（不影响语义），从而修复JSON结构。这比简单正则更可靠。
    try:
        fixed2 = _fix_inner_quotes(fixed)
        return json.loads(fixed2)
    except (json.JSONDecodeError, Exception):
        pass
    # DOC-END id=extract_json_array/fix_inner_quotes#1

    # --- Level 4: 正则兜底提取 ---
    # DOC-BEGIN id=extract_json_array/regex_fallback#1 type=behavior v=1
    # summary: 当所有JSON解析手段都失败时，使用正则直接匹配"point"字段的值，构造List[dict]返回
    # intent: 最终兜底，保证管线不会因单次LLM输出格式错误而完全中断。可能丢失非point字段，但当前管线仅使用point字段。
    print(f"[WARN] All JSON parse attempts failed, falling back to regex extraction. Raw:\n{sliced[:500]}")
    pattern = r'"point"\s*:\s*"((?:[^"\\]|\\.)*)"'
    matches = re.findall(pattern, fixed)
    if matches:
        return [{"point": m.replace('\\"', '"')} for m in matches]
    # DOC-END id=extract_json_array/regex_fallback#1

    raise ValueError(f"Failed to extract any JSON points from LLM output: {sliced[:300]}")
# DOC-END id=extract_json_array#1


# DOC-BEGIN id=fix_inner_quotes#1 type=function v=1
# summary: 接收一个可能包含嵌套双引号的JSON字符串，定位每个"point": "..."值区间内部的多余双引号，将其替换为中文左/右引号，返回修复后的字符串
# intent: LLM输出如 {"point": "TGFβ controls "epithelial identity""} 中，内嵌的双引号会破坏JSON结构。
#         通过查找 "point" 关键字后的冒号和起始引号，然后向前扫描找到真正的结束引号（后面跟 } 或 ,），
#         将中间所有多余双引号替换。这是启发式方法，对当前简单结构 [{"point":"..."}] 有效。
def _fix_inner_quotes(s: str) -> str:
    result = list(s)
    i = 0
    while i < len(s):
        # 查找 "point" 模式
        idx = s.find('"point"', i)
        if idx == -1:
            break
        # 找到冒号
        colon = s.find(':', idx + 7)
        if colon == -1:
            break
        # 找到值的起始引号
        open_q = s.find('"', colon + 1)
        if open_q == -1:
            break
        # 从 open_q+1 开始，找到值的结束引号
        # 结束引号的特征：后面跟着可选空白 + } 或 ,
        j = open_q + 1
        last_quote = -1
        while j < len(s):
            if s[j] == '\\':
                j += 2  # 跳过转义
                continue
            if s[j] == '"':
                # 检查这个引号后面是否是 }, ] 或 ,（可能有空白）
                rest = s[j+1:].lstrip()
                if rest and rest[0] in ('}', ',', ']'):
                    last_quote = j
                    break
                else:
                    # 这是内嵌的双引号，替换为中文引号
                    result[j] = '\u201c'  # "
            j += 1
        i = (last_quote + 1) if last_quote != -1 else (open_q + 1)
    return ''.join(result)
# DOC-END id=fix_inner_quotes#1

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

    logger.info(f"[BlogGen][{article_id}] Step 1: Extracting L1 points...")
    raw_l1 = llm_main.query(prompt_l1, False)
    logger.info(f"[BlogGen][{article_id}] Step 1: LLM returned {len(raw_l1)} chars")
    l1_items = _extract_json_array(raw_l1)
    logger.info(f"[BlogGen][{article_id}] Step 1: Got {len(l1_items)} L1 points")

    tree = OutlineTree(
        title=article_title,
        children=[NodeL1(point=x["point"]) for x in l1_items if "point" in x]
    )

    # ========================================================
    # Step 2: L1 → L2 points（parallel）
    # ========================================================
    logger.info(f"[BlogGen][{article_id}] Step 2: Expanding {len(tree.children)} L1 nodes to L2...")

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
        arr = _extract_json_array(raw)
        return idx, [NodeL2(point=x["point"]) for x in arr if "point" in x]

    with ThreadPoolExecutor(max_workers=min(config.max_workers_step2, len(tree.children))) as pool:
        futures = [pool.submit(expand_l1, i, n1) for i, n1 in enumerate(tree.children)]
        for fut in as_completed(futures):
            i, children = fut.result()
            tree.children[i].children = children
            logger.info(f"[BlogGen][{article_id}] Step 2: L1[{i}] expanded to {len(children)} L2 nodes")

    logger.info(f"[BlogGen][{article_id}] Step 2: Complete")

    # ========================================================
    # Step 3: L2 → detail markdown（parallel）
    # ========================================================
    total_leaves = sum(len(n1.children) for n1 in tree.children)
    logger.info(f"[BlogGen][{article_id}] Step 3: Writing detail for {total_leaves} leaf nodes...")
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
- 图片通过文章中的类似于figure 1B这种, 你就说成[[FIG:1]]中的B图, 任何类似于figure S1B这种在附录中的图片, 你无需引用, 只考虑正文中的图片
- 图片必须用 [[FIG:x]] 占位, 注意, 任何图片类似于[[FIG:1B]]这种是不能接受的, 必须写成[[FIG:1]], 然后你在引用的时候, 说明是哪个子图被你引用, 以及这张子图到底讲述了什么东西, 是什么图, 怎么看(什么东西代表什么东西, 如果无法从文章中推断那就算了)等等
- 定义关键概念，逻辑自洽
- 使用中文
- 建议 400–900 字，不超过 {config.max_leaf_chars}

请开始：
"""
        md = llm.query(prompt, False)
        return i, j, md

    tasks = []
    for i, n1 in enumerate(tree.children):
        for j, n2 in enumerate(n1.children):
            tasks.append((i, j, n1, n2))

    with ThreadPoolExecutor(max_workers=min(config.max_workers_step3, len(tasks))) as pool:
        futures = [pool.submit(write_leaf, *t) for t in tasks]
        done_count = 0
        for fut in as_completed(futures):
            i, j, md = fut.result()
            tree.children[i].children[j].detail_markdown = md
            done_count += 1
            logger.info(f"[BlogGen][{article_id}] Step 3: Leaf [{i}][{j}] done ({done_count}/{total_leaves})")

    logger.info(f"[BlogGen][{article_id}] Step 3: Complete")

    # ========================================================
    # Step 4: Merge L2 → L1 section（parallel）
    # ========================================================
    logger.info(f"[BlogGen][{article_id}] Step 4: Merging {len(tree.children)} sections...")
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
- 使用中文
- 图片通过文章中的类似于figure 1B这种, 你就说成[[FIG:1]]中的B图, 任何类似于figure S1B这种在附录中的图片, 你无需引用, 只考虑正文中的图片
- 文章中的需要用图解释的地方, 图片必须用 [[FIG:x]] 占位, 注意, 任何图片类似于[[FIG:1B]]这种是不能接受的, 必须写成[[FIG:1]], 然后你在引用的时候, 说明是哪个子图被你引用, 以及这张子图到底讲述了什么东西, 是什么图, 怎么看(什么东西代表什么东西, 如果无法从文章中推断那就算了)等等
"""
        llm_raw = llm.query(prompt, False)
        return i, llm_raw

    with ThreadPoolExecutor(max_workers=min(config.max_workers_step4, len(tree.children))) as pool:
        futures = [pool.submit(merge_section, i, n1) for i, n1 in enumerate(tree.children)]
        for fut in as_completed(futures):
            i, sec = fut.result()
            tree.children[i].section_markdown = sec
            logger.info(f"[BlogGen][{article_id}] Step 4: Section [{i}] merged")

    logger.info(f"[BlogGen][{article_id}] Step 4: Complete")

    # ========================================================
    # Step 5: Final merge
    # ========================================================
    logger.info(f"[BlogGen][{article_id}] Step 5: Final merge...")
    sections = "\n\n---\n\n".join(n.section_markdown or "" for n in tree.children)

    prompt_final = f"""
全文上下文：
{full_context}

章节内容：
{sections}

请将以下章节整合为一篇完整博客(Markdown)

要求：
- 标题：# {article_title}
- 开头 TL;DR (5-8 条), 注意请详细讲述清楚概念, 一些知识需要详细讲述清楚
- 图片通过文章中的类似于figure 1B这种, 你就说成[[FIG:1]]中的B图, 任何类似于figure S1B这种在附录中的图片, 你无需引用, 只考虑正文中的图片
- 文章中的需要用图解释的地方, 必须使用 [[FIG:x]] 占位, 注意, 任何图片类似于[[FIG:1B]]这种是不能接受的, 必须写成[[FIG:1]], 然后你在引用的时候, 说明是哪个子图被你引用, 以及这张子图到底讲述了什么东西, 是什么图, 怎么看(什么东西代表什么东西, 如果无法从文章中推断那就算了)等等
- 使用中文
- 讲清楚文章的背景内容, 基本概念, 得到的结论等关键信息
- 合并重复内容, 你最后需要给出的是一个讲解清楚的博客, 需要有你自己的逻辑链条
- 最后给 Figure Index 部分, 每张图必须引用 [[FIG:x]] (这样我才能看得见), 每张图请完整说明所有子图都是什么意思(注意是所有子图都详细说明), 比如A, B, C, ...
- Figure Index 必须包含所有正文图片解释, 也就是figure 1B, figure 2C这些东西, 且一定注意, 包含所有文章正文图片, 一张也不能少
- 任何数学公式, **不要使用()或者[], 正确的使用方法是$$, 一个例子是不要(\\A_i\\), (A_i), 而是$A_i$**
"""
    blog_md = llm_main.query(prompt_final, False)
    logger.info(f"[BlogGen][{article_id}] Step 5: Complete, blog length={len(blog_md)} chars")

    return {
        "blog_markdown": blog_md,
        "used_figs": sorted(set(_extract_fig_placeholders(blog_md))),
        "tree": tree.model_dump(),
    }
