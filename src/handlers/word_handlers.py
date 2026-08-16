# src/handlers/word_handlers.py
import uuid
import threading
import random
import requests
from typing import Optional
from src.services.word_memory import WordMemoryStore
from src.models.word_requests import (
    WordGetNewReq,
    WordGetReviewReq,
    WordGetMasteredReq,
    WordUpdateProgressReq,
    WordGenerateMetaReq,
    WordGetIgnoredReq,
)
import logging
from src.services.llm import LLM

logger = logging.getLogger(__name__)

# 元数据生成任务全局存储，key为task_id
_meta_tasks: dict[str, dict] = {}
# 任务状态读写锁
_meta_lock = threading.Lock()

def _store(user_id: str = "default") -> WordMemoryStore:
    return WordMemoryStore(user_id)


async def handle_word_get_new(req: WordGetNewReq, user_id: str = "default"):
    store = _store(user_id)
    words = store.get_new_words(req.count, req.offset, req.lang)
    return {"ok": True, "words": words}


async def handle_word_get_review(req: WordGetReviewReq, user_id: str = "default"):
    store = _store(user_id)
    words = store.get_review_words(req.count, req.lang)
    return {"ok": True, "words": words}


async def handle_word_get_mastered(req: WordGetMasteredReq, user_id: str = "default"):
    store = _store(user_id)
    words = store.get_mastered_words(req.count, req.lang)
    return {"ok": True, "words": words}


async def handle_word_get_ignored(req: WordGetIgnoredReq, user_id: str = "default"):
    store = _store(user_id)
    words = store.get_ignored_words(req.count, req.lang)
    return {"ok": True, "words": words}


async def handle_word_update_progress(req: WordUpdateProgressReq, user_id: str = "default"):
    store = _store(user_id)
    result = store.update_progress(req.word, req.action)
    return {"ok": True, **result}

# DOC-BEGIN id=handler/word/generate-meta-task#1 type=behavior v=1
# summary: 单词元数据生成后台任务，逐词调用LLM生成解释和例句，更新进度和状态，支持终止
# intent: 后台线程执行避免阻塞网关请求，每处理一个单词更新进度，检测stop_flag立即终止任务，已生成数据会保留
def _meta_generate_task(task_id: str, req: WordGenerateMetaReq, user_id: str):
    store = _store(user_id)
    total_words = len(req.words)
    with _meta_lock:
        _meta_tasks[task_id] = {
            "status": "running",
            "progress": 0,
            "total": total_words,
            "finished": 0,
            "current_word": "",
            "error": "",
            "stop_flag": False
        }
    
    # 初始化LLM实例，指定系统prompt要求严格输出JSON
    llm = LLM(
        api_key=req.api_key,
        llm_url=req.llm_url,
        model_name=req.model_name,
        system_prompt="你是专业的英语词典助手，输出严格遵循JSON格式，不要任何额外解释或markdown格式。",
        format="openai"
    )

    try:
        logger.info(f"[MetaTask {task_id}] Started for user {user_id}, total words: {total_words}")
        for idx, word in enumerate(req.words):
            try:
                # 检查是否要终止任务
                with _meta_lock:
                    if _meta_tasks[task_id]["stop_flag"]:
                        _meta_tasks[task_id]["status"] = "stopped"
                        logger.info(f"[MetaTask {task_id}] Stopped by user request")
                        return
                    _meta_tasks[task_id]["current_word"] = word

                logger.info(f"[MetaTask {task_id}] Processing word {idx+1}/{total_words}: {word}")
                # 构造生成prompt
                prompt = f"""
请为单词 "{word}" 生成如下信息，输出严格是JSON格式，不要任何其他内容：
1. "phonetic_us": 字符串，该单词的美式国际音标，示例格式为 /əˈpl/
2. "cn_explanations": 数组，每个元素是该单词的一个常用中文意思，不同词性、不同用法的释义全部列全，不要限制数量
3. "en_explanations": 数组，每个元素是对应cn_explanations中每个释义的英文解释，顺序和cn_explanations完全一致
4. "examples": 数组，每个元素是一个对象，包含 "en"（英文例句）和 "cn"（对应的中文翻译），共生成5个例句，尽量覆盖更多不同的释义用法，其中至少有1句是日常口语场景的地道表达
"""
                # 调用LLM生成内容
                resp = llm.query(prompt, verbose=False)
                logger.info(f"[MetaTask {task_id}] LLM response for {word}: {resp[:1000]}{'...' if len(resp) > 1000 else ''}")
                # 处理LLM可能返回的markdown代码块标记
                resp = resp.strip()
                if resp.startswith("```json"):
                    resp = resp[7:]
                if resp.endswith("```"):
                    resp = resp[:-3]
                resp = resp.strip()
                # 解析返回的JSON
                import json
                meta = json.loads(resp)

                # DOC-BEGIN id=handler/word/generate-tts-audio#1 type=behavior v=1
                # summary: 为单词和每个例句生成TTS音频，从对应语言的音色列表随机选一个音色，生成的base64音频存入meta
                # intent: TTS生成失败不中断元数据生成流程，无TTS配置时自动跳过；音频格式统一为mp3
                if req.fish_api_key and req.word_tts_reference_ids:
                    try:
                        # 仅支持当前语言音色ID数组格式，无有效配置则跳过TTS生成
                        if not isinstance(req.word_tts_reference_ids, list) or len(req.word_tts_reference_ids) == 0:
                            continue
                        ref_ids = req.word_tts_reference_ids
                        # 调用Fish Audio API生成TTS的通用函数：每次调用随机选择音色
                        def generate_tts(text: str) -> Optional[str]:
                            try:
                                # 每次生成都随机选一个音色
                                selected_ref_id = random.choice(ref_ids)
                                resp = requests.post(
                                    "https://api.fish.audio/v1/tts",
                                    headers={
                                        "Authorization": f"Bearer {req.fish_api_key}",
                                        "Content-Type": "application/json",
                                        "model": "speech-1.6"
                                    },
                                    json={
                                        "text": text,
                                        "reference_id": selected_ref_id,
                                        "format": "mp3"
                                    },
                                    timeout=30
                                )
                                resp.raise_for_status()
                                # 返回base64编码的音频
                                import base64
                                return base64.b64encode(resp.content).decode("utf-8")
                            except Exception as e:
                                logger.warning(f"[MetaTask {task_id}] TTS generate failed for text '{text}': {str(e)}")
                                return None
                        
                        # 生成单词本身的音频
                        meta["audio"] = generate_tts(word)
                        # 生成每个例句的音频
                        if "examples" in meta and isinstance(meta["examples"], list):
                            for ex in meta["examples"]:
                                if "en" in ex:
                                    ex["audio"] = generate_tts(ex["en"])
                    except Exception as e:
                        logger.warning(f"[MetaTask {task_id}] TTS process failed for word {word}: {str(e)}")
                # DOC-END id=handler/word/generate-tts-audio#1

                # 保存到用户单词存储
                store.update_meta(word, meta)
                logger.info(f"[MetaTask {task_id}] Meta saved successfully for word {word}")

                # 更新任务进度
                with _meta_lock:
                    _meta_tasks[task_id]["finished"] = idx + 1
                    _meta_tasks[task_id]["progress"] = int((idx + 1) / total_words * 100)
            except Exception as e:
                logger.error(f"[MetaTask {task_id}] Failed to process word {word}: {str(e)}", exc_info=True)
                # 跳过失败的单词，继续处理下一个
                continue
        
        # 标记任务完成
        logger.info(f"[MetaTask {task_id}] Finished successfully, processed {total_words} words")
        with _meta_lock:
            _meta_tasks[task_id]["status"] = "finished"
    except Exception as e:
        logger.error(f"[MetaTask {task_id}] Global task failed: {str(e)}", exc_info=True)
        with _meta_lock:
            _meta_tasks[task_id]["status"] = "failed"
            _meta_tasks[task_id]["error"] = str(e)
# DOC-END id=handler/word/generate-meta-task#1

# DOC-BEGIN id=handler/word/handle-generate-meta#1 type=api v=1
# summary: 启动单词元数据生成任务，返回任务ID，后台异步执行
# intent: 立即返回任务ID，不等待生成完成，前端通过任务ID轮询查询进度
async def handle_word_generate_meta(req: WordGenerateMetaReq, user_id: str = "default"):
    task_id = str(uuid.uuid4())
    logger.info(f"Received meta generation request: task_id={task_id}, user={user_id}, words_count={len(req.words)}, model={req.model_name}")
    # 启动后台守护线程执行生成任务
    thread = threading.Thread(target=_meta_generate_task, args=(task_id, req, user_id), daemon=True)
    thread.start()
    return {"ok": True, "task_id": task_id}
# DOC-END id=handler/word/handle-generate-meta#1

# DOC-BEGIN id=handler/word/handle-meta-task-status#1 type=api v=1
# summary: 查询元数据生成任务的状态和进度
# intent: 前端轮询调用获取实时进度，返回进度百分比、当前处理单词、完成数、总数量、状态
async def handle_word_meta_task_status(task_id: str):
    with _meta_lock:
        task = _meta_tasks.get(task_id)
        if not task:
            return {"ok": False, "error": "Task not found"}
        # 过滤内部使用的stop_flag字段，只返回业务需要的字段
        return {"ok": True, **{k: v for k, v in task.items() if k != "stop_flag"}}
# DOC-END id=handler/word/handle-meta-task-status#1

# DOC-BEGIN id=handler/word/handle-stop-meta-task#1 type=api v=1
# summary: 终止正在运行的元数据生成任务
# intent: 设置停止标记，后台任务检测到后立即终止，已生成的元数据会保留不会回滚
async def handle_word_stop_meta_task(task_id: str):
    logger.info(f"Received stop request for meta task {task_id}")
    with _meta_lock:
        task = _meta_tasks.get(task_id)
        if not task:
            logger.warning(f"Stop request for non-existent task {task_id}")
            return {"ok": False, "error": "Task not found"}
        if task["status"] == "running":
            task["stop_flag"] = True
            logger.info(f"Set stop flag for running task {task_id}")
        return {"ok": True}
# DOC-END id=handler/word/handle-stop-meta-task#1