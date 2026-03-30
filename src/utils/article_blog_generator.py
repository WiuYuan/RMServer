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

# DOC-BEGIN id=generate_blog_from_article_tree#3 type=function v=3
# summary: 博客生成主函数，接收文章标题、原文、图片目录和可选的images_b64（多模态图片）。
#   有图片时使用query_multimodal将图片+文本一次性发送给多模态LLM；
#   无图片时回退到纯文本query。每一步都有logger.info输出到主线程。
#   返回 {"blog_markdown", "used_figs", "tree"} 格式统一。
# intent: 多模态方案让LLM直接"看"文章图片（figure+table png），而非依赖文字描述猜测图片内容；
#   PDF路径：extract_images_from_adobe_zip返回figure png + table png + 正文text；
#   HTML路径：extract_images返回base64图片（可能需要缩放）+ 空text（用cleaned_text兜底）。
#   所有logger.info输出到python -m server的终端，便于实时排查。
def generate_blog_from_article_tree(
    *,
    task_id: str,
    article_id: str,
    article_title: str,
    article_text: str,
    image_catalog: List[dict],
    config: BlogGenConfig,
    images_b64: Optional[List[str]] = None,
    tc: Optional[Tool_Calls] = None,
) -> dict:

    llm_main = _new_llm(config)
    image_count = len(images_b64) if images_b64 else 0

    logger.info(f"[BlogGen][{article_id}] === Starting blog generation ===")
    logger.info(f"[BlogGen][{article_id}] Article title: {article_title}")
    logger.info(f"[BlogGen][{article_id}] Text length: {len(article_text)} chars")
    logger.info(f"[BlogGen][{article_id}] Image count: {image_count}")
    logger.info(f"[BlogGen][{article_id}] Model: {config.model_name}")

    full_context = _build_full_context(
        article_title,
        article_text,
        image_catalog,
        config.max_article_chars,
    )

    # DOC-BEGIN id=generate_blog_from_article_tree/prompt#1 type=behavior v=2
    # summary: 博客生成提示词，同时适用于多模态和纯文本两种模式。
    #   多模态模式下图片会随提示词一起发送，LLM可直接观看图片内容；
    #   纯文本模式下LLM只能依赖文字描述。
    #   图片索引从1开始对应images_b64列表顺序（也对应extract_images中的index）。
    # intent: 提示词统一，区别仅在于是否有图片输入；
    #   要求LLM用[[FIG:x]]占位引用图片，便于前端替换为实际图片；
    #   强调数学公式用$$而非()，避免渲染问题。
    if image_count > 0:
        prompt = f"""
文章标题：{article_title}

文章正文：
{article_text[:config.max_article_chars]}

以下是文章中的所有图片（共{image_count}张，包括Figure和Table的截图），请仔细观看每张图片的内容。

请根据文章正文和图片内容，写一篇完整的中文博客(Markdown)。

要求：
- 标题：# {article_title}
- 开头 TL;DR (5-8 条)，详细讲述清楚概念
- 当文中需要引用图片时，使用 [[FIG:x]] 占位（x从1开始，对应你看到的图片顺序）
- 对每张图片的子图（如A、B、C）都需要详细说明含义（哪条线代表什么、横纵坐标是什么等）
- 使用中文
- 讲清楚文章的背景、基本概念、结论等关键信息
- 合并重复内容，给出有逻辑链条的讲解
- 最后必须有 Figure Index 部分，每张图用 [[FIG:x]] 引用，详细说明所有子图含义
- 包含所有图片，一张也不能少
- 数学公式用 $...$ 或 $$...$$，不要用 () 或 []
"""
    else:
        prompt = f"""
文章标题：{article_title}

文章正文：
{article_text[:config.max_article_chars]}

请根据上述文章直接写一篇完整的中文博客(Markdown)。

要求：
- 标题：# {article_title}
- 开头 TL;DR (5-8 条)，详细讲述清楚概念
- 文章中的图片引用使用 [[FIG:x]] 占位
- 使用中文
- 讲清楚文章的背景、基本概念、结论等关键信息
- 合并重复内容，给出有逻辑链条的讲解
- 最后给 Figure Index 部分，每张图用 [[FIG:x]] 引用
- 数学公式用 $...$ 或 $$...$$，不要用 () 或 []
"""
    # DOC-END id=generate_blog_from_article_tree/prompt#1

    # DOC-BEGIN id=generate_blog_from_article_tree/llm_call#1 type=behavior v=2
    # summary: 根据是否有图片选择调用方式：有images_b64时用query_multimodal发送图片+文本，
    #   无图片时用query纯文本。两种方式都等待完成后返回完整文本。
    # intent: query_multimodal是非流式的（stream=False），会阻塞直到完整响应返回；
    #   query是流式的（stream=True），边生成边输出。多模态API通常不支持流式。
    if image_count > 0:
        logger.info(f"[BlogGen][{article_id}] Calling multimodal LLM with {image_count} images...")
        blog_md = llm_main.query_multimodal(prompt, images_b64, verbose=True)
        logger.info(f"[BlogGen][{article_id}] Multimodal LLM response length: {len(blog_md)} chars")
    else:
        logger.info(f"[BlogGen][{article_id}] Calling text-only LLM (no images)...")
        blog_md = llm_main.query(prompt, verbose=True)
        logger.info(f"[BlogGen][{article_id}] Text LLM response length: {len(blog_md)} chars")
    # DOC-END id=generate_blog_from_article_tree/llm_call#1

    used_figs = sorted(set(_extract_fig_placeholders(blog_md)))
    logger.info(f"[BlogGen][{article_id}] === Blog generation complete ===")
    logger.info(f"[BlogGen][{article_id}] Blog length: {len(blog_md)} chars, figures used: {used_figs}")

    return {
        "blog_markdown": blog_md,
        "used_figs": used_figs,
        "tree": None,
    }
# DOC-END id=generate_blog_from_article_tree#3
