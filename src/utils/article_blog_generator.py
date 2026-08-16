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

# DOC-BEGIN id=blog-generator/data-models#1 type=design v=2
# summary: 博客生成器的数据模型，仅保留配置类BlogGenConfig；
#   移除了旧版树状结构模型（NodeL1/NodeL2/OutlineTree），因为当前采用一步生成+后处理精修的简化流程
# intent: 简化数据结构，避免维护不再使用的树状分解-回溯合并逻辑；
#   BlogGenConfig保留语言、风格、字符限制等核心配置参数
# DOC-END id=blog-generator/data-models#1


# DOC-BEGIN id=blog-generator/config#1 type=design v=2
# summary: 博客生成配置类，包含LLM调用参数、语言风格设置和字符限制；
#   移除了旧版树状结构参数（l1_points/l2_points/max_workers_*），当前流程为一步生成+后处理精修
# intent: 简化配置，仅保留实际使用的参数；max_article_chars控制输入截断，max_leaf_chars保留用于兼容
class BlogGenConfig(BaseModel):
    model_name: str
    api_key: str
    llm_url: Optional[str] = None

    language: Literal["zh", "en"] = "zh"
    style: Literal["math", "normal", "rigorous"] = "math"

    # DOC-BEGIN id=blog-generator/config-reasoning#1 type=design v=1
    # summary: reasoning_enabled 控制主LLM（博客生成+图片理解）是否启用深度推理模式，
    #   与LLM.__init__的reasoning_enabled参数对应，开启后LLM会调用支持reasoning的模型端点
    # intent: 对应前端ArticleGenerateBlogReq.reasoning_enabled，透传到LLM构造函数
    reasoning_enabled: bool = False
    # DOC-END id=blog-generator/config-reasoning#1

    max_article_chars: int = 120000000
    max_leaf_chars: int = 6000000
# DOC-END id=blog-generator/config#1


# ============================================================
# Helpers
# ============================================================

# DOC-BEGIN id=extract_json_array#1 type=function v=2
# summary: 从LLM原始输出文本中提取JSON数组。依次尝试：(1)直接解析，(2)修复反斜杠后解析，(3)修复未转义双引号后解析，(4)正则逐条提取point字段作为最终兜底。返回List[dict]。
# intent: LLM输出经常包含非法JSON（未转义的反斜杠、嵌入的双引号、多余文本等），单一修复策略不够健壮。采用多级降级策略确保尽可能解析成功，只有完全无法提取时才抛异常。最后的正则兜底可能丢失point以外的字段，但对当前管线足够。
def _extract_json_array(text: str) -> List[dict]:
    """
    Robust JSON array extraction with multi-level fallback.
    现在可以处理任意结构的JSON数组，不再限制于point字段。
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

    # --- Level 3: 修复值内部的未转义双引号 ---
    # DOC-BEGIN id=extract_json_array/fix_inner_quotes#1 type=behavior v=2
    # summary: 逐字符扫描JSON字符串，识别字段值内部的未转义双引号并替换为中文引号，再尝试解析
    # intent: LLM经常输出 "key": "xxx "yyy" zzz" 这类嵌套双引号。通过状态机定位字段值的起止位置，
    #         将中间多余的双引号替换为中文引号（不影响语义），从而修复JSON结构。这比简单正则更可靠。
    try:
        fixed2 = _fix_inner_quotes(fixed)
        return json.loads(fixed2)
    except (json.JSONDecodeError, Exception):
        pass
    # DOC-END id=extract_json_array/fix_inner_quotes#1

    # --- Level 4: 补全不完整JSON ---
    # DOC-BEGIN id=extract_json_array/fix_incomplete_json#1 type=behavior v=1
    # summary: 尝试补全不完整的JSON：修复未闭合的引号、括号，移除尾部垃圾字符
    # intent: LLM输出经常被截断或包含格式错误，导致JSON不完整。
    #   此逻辑尝试最小化修复，使不完整的JSON也能被部分解析。
    fixed3 = _try_fix_incomplete_json(fixed)
    if fixed3 != fixed:
        try:
            return json.loads(fixed3)
        except json.JSONDecodeError:
            pass
    # DOC-END id=extract_json_array/fix_incomplete_json#1

    # --- Level 5: 正则兜底提取（通用字段匹配） ---
    # DOC-BEGIN id=extract_json_array/regex_fallback#1 type=behavior v=3
    # summary: 当所有JSON解析手段都失败时，使用正则逐对象提取JSON对象。
    #   对于Figure Index格式[{...}, {...}]，逐个提取每个对象内的字段；
    #   对于point格式[{point:...}]，提取point字段。
    # intent: 最终兜底，保证管线不会因单次LLM输出格式错误而完全中断。
    #   使用分块正则匹配，确保能处理数组中多个对象的情况。
    logger.warning(f"All JSON parse attempts failed, falling back to regex extraction. Raw:\n{sliced[:500]}")
    return _regex_fallback_extract(sliced)
    # DOC-END id=extract_json_array/regex_fallback#1
# DOC-END id=extract_json_array#1


# DOC-BEGIN id=try_fix_incomplete_json#1 type=function v=1
# summary: 尝试修复不完整的JSON字符串：补全缺失的引号、括号，移除尾部无效字符
# intent: LLM输出经常被截断或格式不完整，此函数尝试最小化修复使其能被解析
def _try_fix_incomplete_json(text: str) -> str:
    """
    尝试修复不完整的JSON字符串。
    处理情况：未闭合引号、缺失的]或}、尾部多余字符。
    """
    if not text:
        return text
    
    result = text.rstrip()
    
    # 移除尾部的非JSON字符（如逗号后面没有内容）
    result = re.sub(r',\s*$', '', result)
    
    # 统计引号、括号数量
    in_string = False
    escape_next = False
    depth_square = 0  # []
    depth_curly = 0   # {}
    last_open_quote = -1
    
    for i, ch in enumerate(result):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            if in_string:
                last_open_quote = i
            continue
        if not in_string:
            if ch == '[':
                depth_square += 1
            elif ch == ']':
                depth_square -= 1
            elif ch == '{':
                depth_curly += 1
            elif ch == '}':
                depth_curly -= 1
    
    # 如果引号未闭合，在末尾添加引号
    if in_string and last_open_quote >= 0:
        result += '"'
    
    # 补全缺失的括号
    while depth_curly > 0:
        result += '}'
        depth_curly -= 1
    while depth_square > 0:
        result += ']'
        depth_square -= 1
    
    return result
# DOC-END id=try_fix_incomplete_json#1


# DOC-BEGIN id=regex_fallback_extract#1 type=function v=1
# summary: 正则兜底提取，逐对象从JSON字符串中提取字段，支持point和Figure Index两种格式
# intent: 当JSON解析完全失败时，使用正则表达式逐个提取对象，确保能处理多对象数组
def _regex_fallback_extract(text: str) -> List[dict]:
    """
    正则兜底提取JSON对象。
    支持两种格式：
    1. point格式: [{"point": "..."}, {"point": "..."}]
    2. Figure Index格式: [{"fig_num": 1, "regions": [...]}, ...]
    """
    results = []
    
    # 策略1: 尝试逐对象匹配 {...}
    # 匹配独立的JSON对象（以{开头，以}结尾）
    object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    object_matches = re.findall(object_pattern, text)
    
    for obj_str in object_matches:
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict):
                results.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        
        # 正则提取该对象内的字段
        field_pattern = r'"([^"]+)"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(\d+)|(\[[^\]]*\]))'
        fields = re.findall(field_pattern, obj_str)
        if fields:
            obj = {}
            for field_name, str_val, num_val, arr_val in fields:
                if str_val:
                    obj[field_name] = str_val.replace('\\"', '"').replace('\\n', '\n')
                elif num_val:
                    try:
                        obj[field_name] = int(num_val) if '.' not in num_val else float(num_val)
                    except ValueError:
                        obj[field_name] = num_val
                elif arr_val:
                    obj[field_name] = arr_val
            if obj:
                results.append(obj)
    
    # 策略2: 如果没有匹配到对象，尝试匹配 point 字段
    if not results:
        point_pattern = r'"point"\s*:\s*"((?:[^"\\]|\\.)*)"'
        point_matches = re.findall(point_pattern, text)
        for point_val in point_matches:
            results.append({"point": point_val.replace('\\"', '"').replace('\\n', '\n')})
    
    return results if results else []
# DOC-END id=regex_fallback_extract#1


# DOC-BEGIN id=fix_inner_quotes#1 type=function v=2
# summary: 接收一个可能包含嵌套双引号的JSON字符串，定位每个字段值区间内部的多余双引号，
#   将其替换为中文左/右引号，返回修复后的字符串。支持任意字段名。
# intent: LLM输出如 {"text": "TGFβ controls "epithelial identity""} 中，内嵌的双引号会破坏JSON结构。
#   通过查找字段名后的冒号和起始引号，向前扫描找到真正的结束引号（后面跟 } 或 , 或 ]），
#   将中间所有多余双引号替换。这是启发式方法，对当前简单结构有效。
def _fix_inner_quotes(s: str) -> str:
    """
    修复JSON字符串中字段值内部的未转义双引号。
    支持任意字段名，不再仅限于"point"字段。
    """
    result = list(s)
    i = 0
    while i < len(s):
        # 查找任意字段名模式："xxx": "..."
        match = re.search(r'"([^"]+)"\s*:\s*"', s[i:])
        if not match:
            break
        
        # 计算值的起始引号位置
        colon_pos = i + match.end() - 1
        
        # 找到值的起始引号
        open_q = colon_pos + 1
        while open_q < len(s) and s[open_q] != '"':
            open_q += 1
        
        if open_q >= len(s):
            break
        
        # 从 open_q+1 开始，找到值的结束引号
        # 结束引号的特征：后面跟着可选空白 + } 或 , 或 ]
        j = open_q + 1
        last_quote = -1
        while j < len(s):
            if s[j] == '\\':
                j += 2  # 跳过转义
                continue
            if s[j] == '"':
                # 检查这个引号后面是否是 }, ] 或 ,（可能有空白）
                rest = s[j+1:].lstrip()
                if not rest or rest[0] in ('}', ',', ']'):
                    last_quote = j
                    break
                else:
                    # 这是内嵌的双引号，替换为中文引号
                    # 根据位置奇偶性选择左/右引号
                    quote_count = sum(1 for k in range(open_q+1, j) if result[k] in ('\u201c', '\u201d'))
                    result[j] = '\u201c' if quote_count % 2 == 0 else '\u201d'
            j += 1
        
        # 移动到下一个可能的位置
        i = (last_quote + 1) if last_quote != -1 else (open_q + 1)
    return ''.join(result)
# DOC-END id=fix_inner_quotes#1

# DOC-BEGIN id=blog-generator/helpers#1 type=function v=1
# summary: 辅助函数集合，包括JSON数组提取、样式规则、图片占位符提取等；
#   这些函数被generate_blog_from_article_tree调用，提供格式化和解析支持
# intent: 将通用逻辑抽取为独立函数，保持主函数简洁；_json_array_rule提供LLM输出格式约束，
#   _style_rules根据配置返回不同的写作风格指导
def _json_array_rule() -> str:
    if style == "math":
        return "偏数学化：定义符号、强调假设、给出逻辑链。"
    if style == "rigorous":
        return "措辞严格，区分假设、证据、结论和局限。"
    return "正常技术博客风格，清晰直接。"


def _extract_fig_placeholders(md: str) -> List[str]:
    if not md:
        return []
    return re.findall(r"\[\[FIG:([0-9]+[a-z]?)\]\]", md)


# DOC-BEGIN id=blog-generator/new-llm-reasoning#1 type=behavior v=1
# summary: _new_llm创建LLM实例时，透传BlogGenConfig.reasoning_enabled到LLM构造函数；
#   该参数控制是否启用深度推理模式，影响多模态博客生成和Figure Index生成的LLM调用
# intent: 将BlogGenConfig中的reasoning_enabled传递到LLM底层，由LLM决定如何启用推理模式
def _new_llm(cfg: BlogGenConfig) -> LLM:
    return LLM(
        api_key=cfg.api_key,
        llm_url=cfg.llm_url,
        model_name=cfg.model_name,
        format="openai",
        ec=None,
        reasoning_enabled=cfg.reasoning_enabled,
    )
# DOC-END id=blog-generator/new-llm-reasoning#1


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

# DOC-BEGIN id=generate_blog_from_article_tree#4 type=function v=4
# summary: 博客生成主函数，接收文章标题、原文、图片目录和可选的images_b64（多模态图片）。
#   有图片时使用query_multimodal将图片+文本一次性发送给多模态LLM；
#   无图片时回退到纯文本query。每一步都有logger.info输出到主线程。
#   返回 {"blog_markdown", "used_figs"} 格式统一。
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

    # DOC-BEGIN id=generate_blog_from_article_tree/prompt#2 type=behavior v=3
    # summary: 博客生成提示词，同时适用于多模态和纯文本两种模式。
    #   多模态模式下图片按顺序发送，每张图片对应一个FIG编号（从1开始）；
    #   允许模型忽略噪音图片（无关装饰图、logo等），只引用对理解文章有帮助的图片。
    # intent: 告诉模型每张图片的编号，方便用[[FIG:x]]引用；
    #   不再要求"包含所有图片"，因为部分图片可能是噪音，强行引用反而影响质量。
    if image_count > 0:
        prompt = f"""
文章标题：{article_title}

文章正文：
{article_text[:config.max_article_chars]}

以下是文章中的所有图片（共{image_count}张，按顺序编号为 FIG 1 到 FIG {image_count}）。
请仔细观看每张图片的内容。

请根据文章正文和图片内容，写一篇完整的中文博客(Markdown)。

要求：
- 标题：# {article_title}
- 开头 TL;DR (5-8 条)，详细讲述清楚概念
- 你将看到 {image_count} 张图片，按顺序是 FIG 1, FIG 2, ..., FIG {image_count}，这个顺序与文章内的出现顺序完全一致：所有Figure按原文出现顺序排列，所有Table按原文出现顺序排在所有Figure之后
- 其中部分图片可能是噪音（如装饰性图标、页眉页脚、logo等与文章内容无关的图片）
- 请只在博客中引用你认为对理解文章有帮助的图片和表格, 无用的噪音请不要出现
- 引用格式：[[FIG:x]]，x是图片编号, 对于表格图片, 你依然采用[[FIG:y]], 而不是[[TABLE:y]], 也就是一视同仁当作图片(当然也可能没有表格)
- 对引用的图片, 详细说明其含义（包括子图A/B/C等、横纵坐标、线条含义等）
- 关于公式图片：如果图片是公式（如数学方程、推导过程），这不是噪音，但也无需用[[FIG:x]]引用。
  请仔细阅读公式图片, 以及文章中存在的任何公式，对于你认为有意义的公式, 请使用$$...$$格式在正文中重写公式，并详细解释每个符号的含义和推导逻辑。
  你需要把公式推导清楚
- 使用中文
- 讲清楚文章的背景、基本概念、结论等关键信息
- 对于关键术语, 请你详细解释清楚, 读者拥有强数学功底, 用公式可以让读者更好理解
- 表格结果需要对比清楚并引用
- 能详细讲就详细讲, blog整体需要很长, 每一个点都细细地讲清楚
- 合并重复内容，给出有逻辑链条的讲解
- 无需生成任何图片的详细索引或Figure Index章节，仅保留博客中所有的[[FIG:x]]图片引用即可，后续会单独生成图片索引
- 数学公式用 $...$ 或 $$...$$，不要用 () 或 [], 且$...$前后都加上空格, $$...$$一定要换行, 以下是例子
 $A_i$ 
$$
A_i
$$
"""
    else:
        prompt = f"""
文章标题：{article_title}

文章正文：
{article_text[:config.max_article_chars]}

请根据文章正文，写一篇完整的中文博客(Markdown)。

要求：
- 标题：# {article_title}
- 开头 TL;DR (5-8 条)，详细讲述清楚概念
- 使用中文
- 讲清楚文章的背景、基本概念、结论等关键信息
- 对于关键术语, 请你详细解释清楚, 读者拥有强数学功底, 用公式可以让读者更好理解
- 能详细讲就详细讲, blog整体需要长一点
- 合并重复内容，给出有逻辑链条的讲解
- 数学公式用 $...$ 或 $$...$$，不要用 () 或 [], 且$...$前后都加上空格, $$...$$一定要换行
"""
    # DOC-END id=generate_blog_from_article_tree/prompt#2

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
    }
# DOC-END id=generate_blog_from_article_tree#4

# DOC-BEGIN id=blog-generator/summary#1 type=documentation v=1
# summary: 博客生成器模块总结
#   该模块实现了一步生成+可选后处理精修的博客生成流程：
#   1. generate_blog_from_article_tree: 主入口函数，接收文章标题、正文、图片，调用多模态LLM生成博客
#   2. 后处理精修由调用方（article_handlers.py）负责，使用单模态LLM对生成的博客进行概念补充和文字润色
#   3. 图片引用格式为[[FIG:x]]，后处理时保持不变
# intent: 简化架构，移除了旧版树状分解-回溯合并逻辑（NodeL1/NodeL2/OutlineTree），
#   采用更直接的一次性生成方式，降低了复杂度同时保持输出质量
# DOC-END id=blog-generator/summary#1
