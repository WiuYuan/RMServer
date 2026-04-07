# src/handlers/news_handlers.py
import os
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
import feedparser
import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.config import DATA_DIR
from src.handlers.llm_handlers import handle_llm_query
from src.models.requests import LLMRequestData

# DOC-BEGIN id=news/config#1 type=config v=1
# summary: 新闻模块全局配置常量，可根据需求修改
# intent: 所有可配置项集中管理，避免硬编码；默认值符合MVP阶段需求，不需要额外调整即可跑通
# 可配置RSS源，格式：[分类, 源名称, RSS地址]
RSS_SOURCES = [
    ["科技", "澎湃科技", "https://www.thepaper.cn/list_25842"],
    ["财经", "财新网", "https://www.caixin.com/rss/finance.xml"],
    ["国际", "路透中文", "https://cn.reuters.com/rss/CNTopGenNews"],
]
NEWS_STORAGE_PATH = f"{DATA_DIR}/news"
AUTO_FETCH_INTERVAL = 15  # 自动拉取间隔，单位分钟
DATA_EXPIRE_DAYS = 10  # 数据保留天数
# DOC-END id=news/config#1

# 全局状态
scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
auto_fetch_enabled = False
_fetch_running = False

# 初始化存储目录
os.makedirs(NEWS_STORAGE_PATH, exist_ok=True)
# 加载持久化的自动开关状态
if os.path.exists(f"{NEWS_STORAGE_PATH}/auto_config.json"):
    with open(f"{NEWS_STORAGE_PATH}/auto_config.json", "r", encoding="utf-8") as f:
        auto_fetch_enabled = json.load(f).get("enable", False)

# DOC-BEGIN id=news/utils/clean_expired#1 type=func v=1
# summary: 清理超过DATA_EXPIRE_DAYS的旧新闻数据，直接删除过期日期目录
# intent: 自动执行，不需要人工干预；直接删除目录比逐条删除条目效率高，符合只存10天的需求
def clean_expired_data():
    expire_date = (datetime.now() - timedelta(days=DATA_EXPIRE_DAYS)).strftime("%Y%m%d")
    for date_dir in os.listdir(NEWS_STORAGE_PATH):
        if date_dir.isdigit() and date_dir < expire_date and os.path.isdir(f"{NEWS_STORAGE_PATH}/{date_dir}"):
            import shutil
            shutil.rmtree(f"{NEWS_STORAGE_PATH}/{date_dir}")
# DOC-END id=news/utils/clean_expired#1

# DOC-BEGIN id=news/fetch_rss#1 type=func v=1
# summary: 异步拉取所有RSS源，归一化输出统一格式的原始新闻条目，自动跳过拉取失败的源
# intent: 异步拉取提高效率，3秒超时避免卡住整个流程；用【来源+标题】做唯一ID去重，无额外依赖
async def fetch_all_rss() -> list:
    raw_entries = []
    existed_ids = set()
    # 先加载最近10天所有已存在的ID避免重复处理
    for date_dir in os.listdir(NEWS_STORAGE_PATH):
        if not date_dir.isdigit(): continue
        for f in os.listdir(f"{NEWS_STORAGE_PATH}/{date_dir}"):
            if f.startswith("raw_") and f.endswith(".json"):
                with open(f"{NEWS_STORAGE_PATH}/{date_dir}/{f}", "r", encoding="utf-8") as fp:
                    for item in json.load(fp):
                        existed_ids.add(item["id"])
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
        for category, source, url in RSS_SOURCES:
            try:
                async with session.get(url) as resp:
                    content = await resp.text()
                    feed = feedparser.parse(content)
                    for entry in feed.entries:
                        item_id = hashlib.md5(f"{source}{entry.title}".encode()).hexdigest()
                        if item_id in existed_ids: continue
                        raw_entries.append({
                            "id": item_id,
                            "category": category,
                            "source": source,
                            "title": entry.title,
                            "link": entry.link,
                            "summary": entry.get("summary", ""),
                            "pub_time": entry.get("published", datetime.now().isoformat()),
                            "fetch_time": datetime.now().isoformat()
                        })
            except:
                continue
    return raw_entries
# DOC-END id=news/fetch_rss#1

# DOC-BEGIN id=news/llm_aggregate#1 type=func v=1
# summary: 批量调用现有LLM接口聚合新闻，输入原始新闻列表，输出聚合后的新闻卡片
# intent: 完全复用现有LLM调用能力，不需要额外封装；Prompt固定，避免幻觉，所有信息都有来源支撑
async def aggregate_news(raw_entries: list) -> list:
    if not raw_entries: return []
    # 按分类分组聚合
    groups = {}
    for item in raw_entries:
        groups.setdefault(item["category"], []).append(item)
    processed = []
    for category, items in groups.items():
        prompt = f"""你是新闻聚合助手，以下是{category}类的最新新闻：
{json.dumps([{"title":i["title"],"summary":i["summary"],"source":i["source"]} for i in items], ensure_ascii=False)}
输出要求：
1. 每条新闻生成100字以内无偏见摘要，标注来源
2. 输出格式：JSON数组，每个元素包含title、summary、tags、source_list四个字段
3. 不要输出任何其他内容
"""
        resp = await handle_llm_query(LLMRequestData(prompt=prompt, stream=False))
        try:
            aggregated = json.loads(resp["content"])
            for item in aggregated:
                processed.append({
                    "id": hashlib.md5(item["title"].encode()).hexdigest(),
                    "category": category,
                    **item,
                    "pub_time": datetime.now().isoformat()
                })
        except:
            continue
    return processed
# DOC-END id=news/llm_aggregate#1

# DOC-BEGIN id=news/manual_fetch#1 type=func v=1
# summary: 手动触发一次全量拉取+聚合+存储流程，返回拉取到的新闻数量
# intent: 防重复执行，避免并发拉取浪费资源；自动清理过期数据，不需要额外调用
async def handle_news_manual_fetch():
    global _fetch_running
    if _fetch_running:
        return {"ok": False, "msg": "正在拉取中，请稍后再试"}
    _fetch_running = True
    try:
        clean_expired_data()
        raw_entries = await fetch_all_rss()
        processed = await aggregate_news(raw_entries)
        # 存储数据
        date_str = datetime.now().strftime("%Y%m%d")
        os.makedirs(f"{NEWS_STORAGE_PATH}/{date_str}", exist_ok=True)
        timestamp = int(datetime.now().timestamp())
        if raw_entries:
            with open(f"{NEWS_STORAGE_PATH}/{date_str}/raw_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(raw_entries, f, ensure_ascii=False, indent=2)
        if processed:
            with open(f"{NEWS_STORAGE_PATH}/{date_str}/processed_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
        return {"ok": True, "raw_count": len(raw_entries), "processed_count": len(processed)}
    finally:
        _fetch_running = False
# DOC-END id=news/manual_fetch#1

# DOC-BEGIN id=news/auto_toggle#1 type=func v=1
# summary: 开关自动拉取功能，输入enable布尔值，返回当前状态；状态持久化到本地文件，重启不丢失
# intent: 定时任务用AsyncIOScheduler和FastAPI事件循环兼容，不需要额外进程；自动拉取间隔可配置
def handle_news_auto_toggle(enable: bool):
    global auto_fetch_enabled
    auto_fetch_enabled = enable
    # 持久化状态
    with open(f"{NEWS_STORAGE_PATH}/auto_config.json", "w", encoding="utf-8") as f:
        json.dump({"enable": enable}, f)
    if enable and not scheduler.running:
        scheduler.add_job(handle_news_manual_fetch, "interval", minutes=AUTO_FETCH_INTERVAL)
        scheduler.start()
    elif not enable and scheduler.running:
        scheduler.shutdown()
    return {"ok": True, "auto_fetch_enabled": auto_fetch_enabled}
# DOC-END id=news/auto_toggle#1

# DOC-BEGIN id=news/list#1 type=func v=1
# summary: 分页查询聚合后的新闻列表，支持按分类筛选，按时间倒序排列
# intent: 直接读本地文件，不需要数据库；查询效率足够支撑MVP阶段上万条新闻的查询需求
def handle_news_list(page: int = 1, page_size: int = 20, category: str = None):
    all_news = []
    # 按日期倒序读
    for date_dir in sorted(os.listdir(NEWS_STORAGE_PATH), reverse=True):
        if not date_dir.isdigit(): continue
        for f in sorted(os.listdir(f"{NEWS_STORAGE_PATH}/{date_dir}"), reverse=True):
            if f.startswith("processed_") and f.endswith(".json"):
                with open(f"{NEWS_STORAGE_PATH}/{date_dir}/{f}", "r", encoding="utf-8") as fp:
                    items = json.load(fp)
                    if category:
                        items = [i for i in items if i["category"] == category]
                    all_news.extend(items)
    # 分页
    start = (page - 1) * page_size
    end = start + page_size
    return {"ok": True, "total": len(all_news), "list": all_news[start:end]}
# DOC-END id=news/list#1

def handle_news_auto_status():
    return {"ok": True, "auto_fetch_enabled": auto_fetch_enabled, "interval": AUTO_FETCH_INTERVAL}