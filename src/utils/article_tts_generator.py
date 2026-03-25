# src/utils/article_tts_generator.py
# ============================================================
# TTS pipeline: blog → LLM rewrite → Fish Audio API → .mp3
# ============================================================

import logging
import re
import requests
import math
from typing import Optional, List

from pydantic import BaseModel

from src.services.llm import LLM

logger = logging.getLogger(__name__)


# ============================================================
# Config
# ============================================================

# DOC-BEGIN id=tts-gen/config#1 type=design v=1
# summary: TTSGenConfig 包含LLM配置（用于将blog转为TTS文本）和Fish Audio配置
#   （reference_id指定说话人声音，fish_api_key用于认证）；chunk_size控制分段大小
# intent: 前端需要传入fish_api_key和reference_id，因为不同用户可能用不同的API key
#   和不同的说话人；chunk_size=2000是Fish Audio单次请求的推荐上限，
#   过大可能导致超时或截断
class TTSGenConfig(BaseModel):
    # LLM config (for rewriting blog to TTS text)
    model_name: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"

    # Fish Audio config
    fish_api_key: str
    reference_id: str
    fish_model: str = "speech-1.6"

    # Chunking
    chunk_size: int = 2000
# DOC-END id=tts-gen/config#1


# ============================================================
# Step 1: LLM rewrite blog → TTS-friendly Chinese text
# ============================================================

# DOC-BEGIN id=tts-gen/rewrite-for-tts#1 type=function v=1
# summary: 接收文章原文和blog markdown，调用LLM生成适合TTS朗读的中文文本；
#   输出会包含情感标记如[laugh]、[sigh]等，去掉所有图片引用、公式转为口语化表达，
#   去掉markdown格式标记；返回纯文本字符串
# intent: blog markdown包含[[FIG:x]]占位符、LaTeX公式、markdown标题/列表等格式，
#   这些内容直接喂给TTS会导致朗读出"左方括号FIG冒号1右方括号"等噪音；
#   通过LLM重写可以将数学公式转为口语（如"alpha"→"阿尔法"），
#   并添加播客风格的情感标记增强听感
def rewrite_blog_for_tts(
    article_text: str,
    blog_markdown: str,
    config: TTSGenConfig,
) -> str:
    llm = LLM(
        api_key=config.api_key,
        llm_url=config.llm_url,
        model_name=config.model_name,
        format="openai",
        ec=None,
    )

    prompt = f"""你是一位专业的中文科技播客主播。你需要将以下博客文章改写为适合TTS（文字转语音）朗读的中文文本。

## 原始文章摘要（供你理解上下文）：
{article_text[:30000]}

## 博客Markdown内容：
{blog_markdown}

## 改写要求：

### 格式要求：
1. **完全去除**所有Markdown格式（#、**、-、```等）
2. **完全去除**所有图片引用（[[FIG:x]]等）
3. **完全去除**所有LaTeX/数学公式符号，将公式转为口语化中文表达（例如：$\\alpha$ → "阿尔法"，$x^2$ → "x的平方"，$\\sum$ → "求和"）
4. 不要输出任何非朗读内容（如"以下是改写结果"之类的元说明）

### 内容要求：
1. 保留博客的核心逻辑和知识点，但用口语化、自然的中文表达
2. 适当添加过渡语句，让听众能跟上逻辑（如"接下来我们来看看..."、"值得注意的是..."）
3. 适当添加播客风格的情感和节奏标记，可以使用以下标签：
   - [laugh] 轻笑
   - [sigh] 叹气
   - [pause] 停顿
4. 开头要有一个简短的引入（如"大家好，今天我们来聊聊..."）
5. 结尾要有一个简短的总结和结束语
6. 整体风格：专业但不枯燥，像是在和朋友聊天讲解一个有趣的研究

### 长度要求：
- 改写后的文本长度应该在原始博客的60%-80%左右
- 不需要覆盖博客中所有的细枝末节，聚焦核心内容

请直接输出改写后的TTS文本，不要有任何额外说明：
"""

    logger.info("[TTSGen] Calling LLM to rewrite blog for TTS...")
    tts_text = llm.query(prompt, False)
    logger.info(f"[TTSGen] LLM returned TTS text, length={len(tts_text)} chars")

    # DOC-BEGIN id=tts-gen/post-clean#1 type=behavior v=1
    # summary: 对LLM返回的TTS文本做后处理清洗：去除残留的markdown代码块标记、
    #   多余空行压缩为单空行、去除首尾空白
    # intent: LLM有时会忽略指令仍然输出```包裹的代码块或多余的markdown标记，
    #   这些进入TTS会产生噪音；多余空行在TTS中无意义但会增加chunk数量
    tts_text = re.sub(r'```[\s\S]*?```', '', tts_text)
    tts_text = re.sub(r'#{1,6}\s*', '', tts_text)
    tts_text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', tts_text)
    tts_text = re.sub(r'\$+[^$]*\$+', '', tts_text)
    tts_text = re.sub(r'\[\[FIG:\d+[a-z]?\]\]', '', tts_text)
    tts_text = re.sub(r'\n{3,}', '\n\n', tts_text)
    tts_text = tts_text.strip()
    # DOC-END id=tts-gen/post-clean#1

    return tts_text
# DOC-END id=tts-gen/rewrite-for-tts#1


# ============================================================
# Step 2: Split text into chunks
# ============================================================

# DOC-BEGIN id=tts-gen/split-chunks#1 type=function v=1
# summary: 将TTS文本按chunk_size（默认2000字符）分段，优先在句号/问号/感叹号处断句，
#   如果一段内没有句末标点则在最近的逗号/分号处断，最坏情况硬切；返回字符串列表
# intent: Fish Audio API对单次请求文本长度有限制（过长会超时或质量下降），
#   需要分段请求然后拼接音频；在标点处断句可以避免朗读时出现不自然的截断；
#   三级降级策略（句号→逗号→硬切）确保任何输入都能被分段
def split_text_to_chunks(text: str, chunk_size: int = 2000) -> List[str]:
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break

        # 在chunk_size范围内找最后一个句末标点
        window = remaining[:chunk_size]
        cut_pos = -1

        # 优先找句号、问号、感叹号
        for punct in ['。', '！', '？', '；', '\n']:
            pos = window.rfind(punct)
            if pos > cut_pos:
                cut_pos = pos

        # 如果没找到，找逗号、分号
        if cut_pos < chunk_size // 2:
            for punct in ['，', '、', ',', ';']:
                pos = window.rfind(punct)
                if pos > cut_pos:
                    cut_pos = pos

        # 最坏情况硬切
        if cut_pos < chunk_size // 4:
            cut_pos = chunk_size - 1

        chunk = remaining[:cut_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut_pos + 1:].strip()

    return chunks
# DOC-END id=tts-gen/split-chunks#1


# ============================================================
# Step 3: Call Fish Audio API for each chunk, concatenate
# ============================================================

# DOC-BEGIN id=tts-gen/call-fish-audio#1 type=function v=1
# summary: 对文本分段列表逐段调用Fish Audio TTS API，每段返回mp3二进制数据，
#   将所有段的mp3数据直接拼接（mp3格式支持帧级拼接）后返回完整音频字节流
# intent: mp3是基于帧的格式，多个mp3文件直接拼接在大多数播放器中可以正常播放，
#   不需要复杂的音频处理库；逐段调用而非一次性调用是因为Fish Audio API有文本长度限制；
#   每段调用之间记录日志方便追踪哪一段失败；失败时抛出异常由上层处理
def call_fish_audio_tts(
    chunks: List[str],
    config: TTSGenConfig,
) -> bytes:
    all_audio_data = b""

    for i, chunk in enumerate(chunks):
        logger.info(f"[TTSGen] Calling Fish Audio API for chunk {i+1}/{len(chunks)}, "
                     f"length={len(chunk)} chars")

        # DOC-BEGIN id=tts-gen/fish-api-call-detail#1 type=behavior v=1
        # summary: 调用Fish Audio TTS API，headers中包含Authorization和model字段，
        #   body中包含text、reference_id和format；model使用config.fish_model（默认speech-1.6）
        # intent: Fish Audio API要求在headers中传model字段指定合成模型版本（如s2-pro/speech-1.6），
        #   不传会使用服务端默认模型；timeout=120秒因为长文本合成可能较慢
        response = requests.post(
            "https://api.fish.audio/v1/tts",
            headers={
                "Authorization": f"Bearer {config.fish_api_key}",
                "Content-Type": "application/json",
                "model": config.fish_model,
            },
            json={
                "text": chunk,
                "reference_id": config.reference_id,
                "format": "mp3",
            },
            timeout=120,
        )
        # DOC-END id=tts-gen/fish-api-call-detail#1

        if response.status_code != 200:
            error_detail = response.text[:500]
            logger.error(f"[TTSGen] Fish Audio API error on chunk {i+1}: "
                         f"status={response.status_code}, body={error_detail}")
            raise RuntimeError(
                f"Fish Audio API error on chunk {i+1}/{len(chunks)}: "
                f"HTTP {response.status_code} - {error_detail}"
            )

        chunk_audio = response.content
        logger.info(f"[TTSGen] Chunk {i+1}/{len(chunks)} returned {len(chunk_audio)} bytes")
        all_audio_data += chunk_audio

    logger.info(f"[TTSGen] All chunks done, total audio size={len(all_audio_data)} bytes")
    return all_audio_data
# DOC-END id=tts-gen/call-fish-audio#1


# ============================================================
# Full pipeline
# ============================================================

# DOC-BEGIN id=tts-gen/generate-tts#1 type=function v=1
# summary: TTS生成完整管线：(1)LLM将blog+文章重写为TTS文本 → (2)分段 → (3)调用Fish Audio API →
#   返回dict包含tts_text(中间文本)和audio_bytes(mp3音频字节流)
# intent: 将三步封装为单一入口，供tts_handlers.py中的worker调用；
#   返回tts_text是为了可以将中间文本也保存下来（方便调试和前端展示）；
#   audio_bytes由调用方负责写入文件
def generate_tts_from_blog(
    *,
    article_id: str,
    article_text: str,
    blog_markdown: str,
    config: TTSGenConfig,
) -> dict:
    # Step 1: LLM rewrite
    tts_text = rewrite_blog_for_tts(
        article_text=article_text,
        blog_markdown=blog_markdown,
        config=config,
    )

    # Step 2: Split
    chunks = split_text_to_chunks(tts_text, config.chunk_size)
    logger.info(f"[TTSGen][{article_id}] Split into {len(chunks)} chunks")

    # Step 3: Fish Audio TTS
    audio_bytes = call_fish_audio_tts(chunks, config)

    return {
        "tts_text": tts_text,
        "audio_bytes": audio_bytes,
    }
# DOC-END id=tts-gen/generate-tts#1