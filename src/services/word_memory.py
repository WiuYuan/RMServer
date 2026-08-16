# src/services/word_memory.py
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
from wordfreq import top_n_list
from src.config import WORD_MEMORY_ROOT


# ── 单词列表缓存（按语言） ──
_word_lists: dict[str, list[str]] = {}


def ensure_word_list(lang: str = "en", n: int = 50000):
    """
    懒加载并缓存指定语言的常用单词列表（已按频率降序排列）。
    首次调用可能较慢（~1-2s），后续从缓存返回。
    """
    if lang not in _word_lists:
        _word_lists[lang] = list(top_n_list(lang, n))
    return _word_lists[lang]


# ── 用户记忆度持久化 ──
class WordMemoryStore:
    """
    管理单个用户的单词记忆数据。
    
    数据结构示例:
    {
        "apple":  {"memory": 3,  "attempts": 5, "last_update": "2026-04-19T20:00:00"},
        "abandon": {"memory": -2, "attempts": 3, "last_update": "2026-04-19T20:05:00"}
    }
    
    memory 值:
      - 正数: 背诵成功次数净差（越大掌握越好）
      - 负数: 背诵失败次数净差（越小越需要复习）
      - 0:    刚创建记录但尚无成功/失败（也视为需要复习）
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        Path(WORD_MEMORY_ROOT).mkdir(parents=True, exist_ok=True)
        self.filepath = os.path.join(WORD_MEMORY_ROOT, f"{user_id}.json")
        self._data: dict = {}
        self._load()

    # ── 持久化 ──

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def _save(self):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ── 核心操作 ──

    def has_word(self, word: str) -> bool:
        """该用户是否已有此单词的记忆记录"""
        return word in self._data

    def get_memory(self, word: str) -> Optional[int]:
        return self._data.get(word, {}).get("memory")

    def update_progress(self, word: str, action: Literal["master", "known", "unknown", "ignore"]) -> dict:
        """
        更新单词记忆度，返回更新后的状态。
        - success=True:  memory += 1
        - success=False: memory -= 1（第一次忘记）
        """
        if word not in self._data:
            self._data[word] = {"memory": 0, "attempts": 0, "difficulty": 0, "last_shown_at": None, "is_ignored": False, "meta": {}}

        entry = self._data[word]
        entry["attempts"] += 1
        entry["last_shown_at"] = datetime.now().isoformat()
        if action == "master":
            # 标记已掌握：memory设为正，清难度
            entry["memory"] = max(entry["memory"], 1)
            entry["difficulty"] = 0
        elif action == "known":
            # 认识：难度+1，保留在背诵队列
            entry["memory"] = 0
            entry["difficulty"] = max(0, entry["difficulty"] + 1)
        elif action == "unknown":
            # 不认识：难度+3，保留在背诵队列
            entry["memory"] = 0
            entry["difficulty"] = max(0, entry["difficulty"] + 3)
        elif action == "ignore":
            # 标记为无意义：不参与任何排序和复习
            entry["is_ignored"] = True
        elif action == "reset":
            # 重置为从未学习过：仅清空学习状态，保留meta数据
            if word in self._data:
                # 保留meta，重置所有学习相关字段
                self._data[word] = {
                    **self._data[word],
                    "memory": None,
                    "attempts": 0,
                    "difficulty": 0,
                    "last_shown_at": None,
                    "is_ignored": False
                }
                self._save()
            return {"reset": True, "word": word}
        elif action == "restore":
            # 从忽略列表恢复：取消忽略标记，设置为待复习状态
            entry["is_ignored"] = False
            entry["memory"] = 0
            entry["difficulty"] = 1
        entry["last_update"] = datetime.now().isoformat()
        self._save()
        return entry

    # DOC-BEGIN id=service/word-memory/update-meta#1 type=behavior v=1
    # summary: 更新单词的LLM生成元数据，自动覆盖旧数据，持久化到存储，不影响单词的学习状态
    # intent: 元数据是独立附加信息，仅补充解释/例句等内容，不会改变单词的新单词/复习状态
    def update_meta(self, word: str, meta: dict) -> None:
        if word not in self._data:
            # 仅生成元数据的单词无学习记录，默认学习状态为空
            self._data[word] = {"memory": None, "attempts": 0, "difficulty": 0, "last_shown_at": None, "is_ignored": False, "meta": {}}
        self._data[word]["meta"] = meta
        self._save()
    # DOC-END id=service/word-memory/update-meta#1

    def get_new_words(self, count: int, offset: int, lang: str = "en") -> list[str]:
        """
        获取新单词，按频率排名，排除已有记录的单词
        """
        word_list = ensure_word_list(lang)
        # 跳过offset个单词，排除已经有记录的
        new_words = []
        for word in word_list[offset:]:
            # 只要无学习记录（attempts=0）就算新单词，不管有没有生成过元数据
            if not self.has_word(word) or int(self._data[word].get("attempts") or 0) == 0:
                # 跳过已忽略的单词
                if self.has_word(word) and self._data[word].get("is_ignored", False):
                    continue
                new_words.append(word)
                if len(new_words) >= count:
                    break
        # 返回带meta的对象数组，而非纯字符串
        return [{"word": w, "meta": self._data.get(w, {}).get("meta", {})} for w in new_words]

    def get_review_words(self, count: int, lang: str = "en") -> list[str]:
        """
        获取正在背诵的单词（memory=0），count>0时随机取count个，count<=0时返回全部按字母排序
        """
        reviewing = [
            (word, info)
            for word, info in self._data.items()
            if int(info.get("attempts") or 0) > 0 and int(info.get("memory") or 0) == 0 and not info.get("is_ignored", False)
        ]
        if not reviewing:
            return []
        
        if count <= 0:
            # 管理模式：返回全部，按最后更新时间降序排序（最近更新的在前），不更新展示时间
            reviewing.sort(key=lambda x: x[1].get("last_update", ""), reverse=True)
            selected = [w for w, _ in reviewing]
        else:
            # 正常复习模式：随机选取count个，更新展示时间
            random.shuffle(reviewing)
            selected = [w for w, _ in reviewing[:count]]
            # 更新选中复习单词的上次展示时间，实现轮转效果
            for word in selected:
                self._data[word]["last_shown_at"] = datetime.now().isoformat()
            self._save()
        
        # 返回带meta的对象数组，而非纯字符串
        return [{"word": w, "meta": self._data.get(w, {}).get("meta", {})} for w in selected]

    def get_mastered_words(self, count: int, lang: str = "en") -> list[str]:
        """
        获取已掌握的单词：count>0时队列轮转+随机排序，count<=0时返回全部按字母排序
        1. 筛选 memory > 0 的单词
        2. count>0时：按上次展示时间升序排列（越久没展示越优先），取前 2*count 个随机打乱，取前 count 个
        3. count<=0时：按单词字母升序排序，返回全部
        """
        mastered = [
            (word, info)
            for word, info in self._data.items()
            if int(info.get("attempts") or 0) > 0 and int(info.get("memory") or 0) > 0 and not info.get("is_ignored", False)
        ]
        if not mastered:
            return []
        
        if count <= 0:
            # 管理模式：返回全部，按最后更新时间降序排序（最近更新的在前），不更新展示时间
            mastered.sort(key=lambda x: x[1].get("last_update", ""), reverse=True)
            selected = [w for w, _ in mastered]
        else:
            # 正常查看模式：队列轮转+随机排序
            mastered.sort(key=lambda x: x[1].get("last_shown_at") or "")
            pool_size = count * 2
            candidates = [w for w, _ in mastered[:pool_size]]
            random.shuffle(candidates)
            selected = candidates[:count]
            # 更新选中单词的上次展示时间，实现轮转效果
            for word in selected:
                self._data[word]["last_shown_at"] = datetime.now().isoformat()
            self._save()
        
        # 统一返回带meta的对象数组，和其他接口格式一致
        return [{"word": w, "meta": self._data.get(w, {}).get("meta", {})} for w in selected]

    def get_ignored_words(self, count: int, lang: str = "en") -> list[str]:
        """
        获取已标记为忽略的单词：count>0时随机取count个，count<=0时返回全部按最后更新时间降序排序
        """
        ignored = [
            (word, info)
            for word, info in self._data.items()
            if info.get("is_ignored", False)
        ]
        if not ignored:
            return []
        
        if count <= 0:
            # 管理模式：返回全部，按最后更新时间降序排序（最近更新的在前）
            ignored.sort(key=lambda x: x[1].get("last_update", ""), reverse=True)
            selected = [w for w, _ in ignored]
        else:
            random.shuffle(ignored)
            selected = [w for w, _ in ignored[:count]]
        
        # 返回带meta的对象数组，和其他接口格式一致
        return [{"word": w, "meta": self._data.get(w, {}).get("meta", {})} for w in selected]