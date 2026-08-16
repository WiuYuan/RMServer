# src/models/word_requests.py
from pydantic import BaseModel
from typing import Optional, Literal, List, Union

class WordGetNewReq(BaseModel):
    count: int = 10
    offset: int = 0
    lang: str = "en"
    user_id: str = "default"


class WordGetReviewReq(BaseModel):
    count: int = 10
    lang: str = "en"
    user_id: str = "default"


class WordGetMasteredReq(BaseModel):
    count: int = 10
    lang: str = "en"
    user_id: str = "default"


class WordGetIgnoredReq(BaseModel):
    count: int = 10
    lang: str = "en"
    user_id: str = "default"


class WordUpdateProgressReq(BaseModel):
    word: str                # 单词
    action: Literal["master", "known", "unknown", "ignore", "reset", "restore"]  # 操作类型：标记已掌握/认识/不认识/忽略无意义/重置为未学习/从忽略恢复
    lang: str = "en"


class WordGenerateMetaReq(BaseModel):
    words: List[str]  # 需要生成元数据的单词列表
    lang: str = "en"
    user_id: str = "default"
    model_name: str = "qwen3:8b"
    llm_url: str = "http://localhost:11434/api/chat"
    api_key: str = ""
    fish_api_key: Optional[str] = ""  # Fish Audio TTS密钥
    word_tts_reference_ids: Optional[list[str]] = None  # 当前语言的TTS音色ID数组，随机选一个生成发音


class WordMetaTaskStatusReq(BaseModel):
    task_id: str  # 生成任务ID


class WordStopMetaTaskReq(BaseModel):
    task_id: str  # 要停止的生成任务ID