
# src/handlers/news_handlers.py
import os
import json
import hashlib
import asyncio
import shutil
from datetime import datetime, timedelta
import feedparser
import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.config import DATA_DIR
from src.services.llm import LLM
import logging

logger = logging.getLogger(__name__)

# DOC-BEGIN id=news/config#1 type=config v=1
# summary: 新闻模块全局配置常量，可根据需求修改
# intent: 所有可配置项集中管理，避免硬编码；默认值符合MVP阶段需求，不需要额外调整即可跑通
# 可配置RSS源，格式：[分类, 源名称, RSS地址]
RSS_SOURCES = [
    ["Tech", "TechCrunch", "https://techcrunch.com/feed/"],
    ["Tech", "The Verge", "https://www.theverge.com/rss/index.xml"],
    ["Tech", "Wired", "https://www.wired.com/feed/rss"],
    ["Tech", "Ars Technica", "https://arstechnica.com/feed/"],
    ["Tech", "Hacker News", "https://news.ycombinator.com/rss"],
    ["Tech", "Engadget", "https://www.engadget.com/rss.xml"],
    ["Finance", "Yahoo Finance", "https://finance.yahoo.com/news/rssindex"],
    ["Finance", "MarketWatch Top Stories", "http://feeds.marketwatch.com/marketwatch/topstories/"],
    ["Global", "BBC World News", "https://feeds.bbci.co.uk/news/world/rss.xml"],
    ["Global", "NYT World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"],
    ["Global", "The Guardian World", "https://www.theguardian.com/world/rss"],
    ["Global", "NPR News", "https://feeds.npr.org/1001/rss.xml"],
]
NEWS_STORAGE_PATH = f"{DATA_DIR}/news"
DATA_PATH = f"{NEWS_STORAGE_PATH}"  # 新：HTML存储根目录（每日期文件夹下按source分目录存放html）
AUTO_FETCH_INTERVAL = 15  # 自动拉取间隔，单位分钟
DATA_EXPIRE_DAYS = 10  # 数据保留天数
HUMAN_PREFERENCE_PATH = f"{NEWS_STORAGE_PATH}/human_preference.json"  # 独立人类偏好存储，不受过期删除影响
# DOC-END id=news/config#1

# 全局状态
scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
auto_fetch_enabled = False
_fetch_running = False

# HTML下载队列：每项形如 {"id": ..., "link": ..., "source": ..., "date_str": ...}
_download_queue = []
_download_running = False  # 防止多个下载worker同时运行

# LLM打分进程全局状态
_scoring_running = False
_total_to_score = 0
_scored_count = 0
_cancel_scoring_flag = False

# 初始化存储目录
os.makedirs(NEWS_STORAGE_PATH, exist_ok=True)
# 加载持久化的自动开关状态
if os.path.exists(f"{NEWS_STORAGE_PATH}/auto_config.json"):
    with open(f"{NEWS_STORAGE_PATH}/auto_config.json", "r", encoding="utf-8") as f:
        auto_fetch_enabled = json.load(f).get("enable", False)
logger.info(f"[News] Loaded initial auto fetch status from local config: {auto_fetch_enabled}")

# DOC-BEGIN id=news/utils/clean_expired#1 type=func v=1
# summary: 清理超过DATA_EXPIRE_DAYS的旧新闻数据，直接删除过期日期目录
# intent: 自动执行，不需要人工干预；直接删除目录比逐条删除条目效率高，符合只存10天的需求
def clean_expired_data(days: int = DATA_EXPIRE_DAYS) -> int:
    expire_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    removed_count = 0
    for date_dir in os.listdir(NEWS_STORAGE_PATH):
        if date_dir.isdigit() and date_dir < expire_date and os.path.isdir(f"{NEWS_STORAGE_PATH}/{date_dir}"):
            import shutil
            shutil.rmtree(f"{NEWS_STORAGE_PATH}/{date_dir}")
            removed_count += 1
    logger.info(f"[News] Cleaned expired news data before {expire_date}, removed {removed_count} expired date directories")
    return removed_count
# DOC-END id=news/utils/clean_expired#1

# DOC-BEGIN id=news/utils/human-preference#1 type=func v=1
# summary: 加载独立存储的人类偏好数据，返回格式为{score: [list of samples]}，每个分数对应最多3个样本
# intent: 独立文件存储不受过期数据清理影响，自动处理文件不存在/格式错误的情况，返回默认空结构
def load_human_preference() -> dict:
    default = {str(s): [] for s in range(1, 11)}
    if not os.path.exists(HUMAN_PREFERENCE_PATH):
        return default
    try:
        with open(HUMAN_PREFERENCE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 合并默认结构，避免缺失分数槽
            for s in default:
                if s not in data:
                    data[s] = []
                # 保证每个槽最多3个样本
                if len(data[s]) > 3:
                    data[s] = data[s][-3:]
            return data
    except:
        return default

# DOC-BEGIN id=news/utils/human-preference-save#1 type=func v=1
# summary: 保存人类偏好数据到独立文件
# intent: 每次修改后立即持久化，避免数据丢失
def save_human_preference(data: dict):
    with open(HUMAN_PREFERENCE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# DOC-END id=news/utils/human-preference-save#1
# DOC-END id=news/utils/human-preference#1

# DOC-BEGIN id=news/delete_old#1 type=func v=1
# summary: 手动删除超过指定天数的旧新闻，输入days正整数，返回删除的目录数量
# intent: 前端手动触发，动态指定删除天数，不需要修改配置；删除后自动刷新列表
# DOC-BEGIN id=news/delete_old#1 type=func v=2
# summary: 手动删除超过指定天数的旧新闻，输入days>=0，0代表删除所有新闻，返回删除的目录数量
# intent: 前端手动触发，支持清空所有新闻需求；删除后自动刷新列表
def handle_news_delete_old(days: int = 3):
    if days < 0:
        return {"ok": False, "error": "Days must be >= 0 (0 = delete all news)"}
    # 0天代表删除所有新闻
    if days == 0:
        expire_date = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
        removed_count = 0
        for date_dir in os.listdir(NEWS_STORAGE_PATH):
            if date_dir.isdigit() and date_dir < expire_date and os.path.isdir(f"{NEWS_STORAGE_PATH}/{date_dir}"):
                shutil.rmtree(f"{NEWS_STORAGE_PATH}/{date_dir}")
                removed_count += 1
        logger.info(f"[News] Deleted all news data, removed {removed_count} date directories")
        return {"ok": True, "removed_count": removed_count}
    removed = clean_expired_data(days)
    return {"ok": True, "removed_count": removed}
# DOC-END id=news/delete_old#1
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
        raw_file = f"{NEWS_STORAGE_PATH}/{date_dir}/raw.json"
        if os.path.exists(raw_file):
            with open(raw_file, "r", encoding="utf-8") as fp:
                for item in json.load(fp):
                    existed_ids.add(item["id"])
    # DOC-BEGIN id=news/fetch_rss/ua-timeout-adjust#1 type=optimization v=1
    # summary: RSS拉取请求增加10秒超时+标准浏览器User-Agent头，避免国外源/反爬策略拦截
    # intent: 解决MarketWatch等RSS源浏览器可访问但代码拉取失败问题；UA与HTML下载逻辑保持一致，降低被识别为爬虫概率
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        for category, source, url in RSS_SOURCES:
            try:
                async with session.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }) as resp:
    # DOC-END id=news/fetch_rss/ua-timeout-adjust#1
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
            except Exception as e:
                logger.warning(f"[News] Failed to fetch RSS source [{source}] <{url}>, error: {str(e)}")
                continue
    logger.info(f"[News] RSS fetch completed, got {len(raw_entries)} new unique entries, skipped {len(existed_ids)} existing duplicates")
    return raw_entries
# DOC-END id=news/fetch_rss#1

# DOC-BEGIN id=news/llm_aggregate#1 type=func v=1
# summary: 批量调用现有LLM接口聚合新闻，输入原始新闻列表，输出聚合后的新闻卡片
# intent: 完全复用现有LLM调用能力，不需要额外封装；Prompt固定，避免幻觉，所有信息都有来源支撑
# DOC-BEGIN id=news/llm_score#2 type=func v=2
# summary: 批量给无打分条目调LLM打分，输入summary+历史人类打分，输出1~10分，逐条更新raw.json
#   新增model_name/api_key/llm_url参数，从调用方传入打分模型配置
# intent: 以历史人类打分作为few-shot参考，让LLM学习用户偏好；每次只处理无打分条目，避免重复调用
#   修复：原来直接调用handle_llm_query没有传入模型配置，现在从参数传入
async def llm_score_entries(date_str: str, model_name: str = None, api_key: str = None, llm_url: str = None) -> tuple:
    global _scoring_running, _total_to_score, _scored_count, _cancel_scoring_flag
    # 防止重复启动打分进程
    if _scoring_running:
        return 0, 0
    raw_file = f"{NEWS_STORAGE_PATH}/{date_str}/raw.json"
    if not os.path.exists(raw_file): return 0, 0
    with open(raw_file, "r", encoding="utf-8") as f:
        entries = json.load(f)
    
    # 初始化打分状态
    _scoring_running = True
    _cancel_scoring_flag = False
    _scored_count = 0
    
    # 读取独立存储的人类偏好作为参考，每个分数最多3个样本，总样本最多30个
    pref = load_human_preference()
    human_examples = []
    # 按分数从高到低排序示例，优先展示高分偏好
    for score in sorted(pref.keys(), key=lambda x: int(x), reverse=True):
        human_examples.extend(pref[score])
    examples_text = ""
    if human_examples:
        examples_text = "\n\n以下是用户历史打分参考，请学习其偏好：\n" + json.dumps(human_examples, ensure_ascii=False)
    
    # 收集需要评分的条目
    entries_to_score = [entry for entry in entries if entry.get("llm_score") is None]
    _total_to_score = len(entries_to_score)
    if not entries_to_score:
        _scoring_running = False
        return 0, 0
    
    # DOC-BEGIN id=news/llm_score_batch#1 type=logic v=1
    # summary: 批量评分逻辑，每批最多20条新闻一起发送给LLM评分，输入为条目列表和人类示例文本，输出评分成功和失败计数
    # intent: 提高效率减少LLM调用次数；使用逗号分隔整数作为响应格式，简单解析；如果解析失败则跳过该批次，不影响其他批次
    batch_size = 20
    scored = 0
    failed = 0
    changed = False
    
    for i in range(0, len(entries_to_score), batch_size):
        batch = entries_to_score[i:i+batch_size]
        # 构建批量提示词
        prompt_lines = ["你是新闻质量评分助手。请根据以下新闻的标题和摘要，为每个新闻打1~10分。", "新闻列表："]
        for idx, entry in enumerate(batch, 1):
            prompt_lines.append(f"{idx}. 标题：{entry['title']}, 摘要：{entry['summary']}, 来源：{entry['source']}")
        prompt_lines.append(f"请输出{len(batch)}个整数，用逗号分隔，例如：5,7,3,8,...")
        prompt_lines.append("只输出分数，不要输出任何其他内容。")
        if examples_text:
            prompt_lines.append(examples_text)
        prompt = "\n".join(prompt_lines)
        
        try:
                # DOC-BEGIN id=news/llm_score/direct-llm-call#1 type=logic v=1
            # summary: 直接调用LLM类执行打分请求，输入prompt文本、模型配置，返回LLM输出字符串
            # intent: 绕过handle_llm_query适配层避免字段不匹配问题；用asyncio.to_thread执行同步query方法防止阻塞FastAPI事件循环；verbose设为false避免打印冗余日志
            llm = LLM(
                api_key=api_key,
                llm_url=llm_url,
                model_name=model_name,
                format="openai"
            )
            content = await asyncio.to_thread(llm.query, prompt, verbose=False)
            content = content.strip()
            # DOC-END id=news/llm_score/direct-llm-call#1
            # 尝试解析逗号分隔的分数
            # 移除可能的方括号或其他字符
            content = content.strip('[]')
            score_strs = [s.strip() for s in content.split(',') if s.strip()]
            scores = []
            for s in score_strs:
                try:
                    score = int(s)
                    if 1 <= score <= 10:
                        scores.append(score)
                    else:
                        scores.append(None)
                except ValueError:
                    scores.append(None)
            # 确保分数数量与批次匹配，如果不足，用None填充
            while len(scores) < len(batch):
                scores.append(None)
            # 检查取消标志
            if _cancel_scoring_flag:
                logger.info(f"[News] Scoring canceled by user, processed {scored} items, remaining {_total_to_score - scored} items skipped")
                break
            
            # 分配分数
            for j, entry in enumerate(batch):
                if scores[j] is not None:
                    entry["llm_score"] = scores[j]
                    scored += 1
                    changed = True
                else:
                    failed += 1
            
            # 更新全局计数 + 输出单批次完成日志
            _scored_count = scored
            logger.info(f"[News] Batch scoring completed: {len(batch)} items processed, total progress: {scored}/{_total_to_score} ({round(scored/_total_to_score*100, 1)}%)")
        except Exception as e:
            logger.warning(f"[News] Failed to score batch starting at index {i}: {e}")
            failed += len(batch)
            continue
    # DOC-END id=news/llm_score_batch#1
    
    if changed:
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
    
    # 重置打分状态
    _scoring_running = False
    _total_to_score = 0
    _scored_count = 0
    _cancel_scoring_flag = False
    
    logger.info(f"[News] LLM scoring completed: scored={scored}, failed={failed}")
    return scored, failed
# DOC-END id=news/llm_score#2
# DOC-END id=news/llm_aggregate#1

# DOC-BEGIN id=news/manual_fetch#2 type=func v=2
# summary: 手动触发一次RSS全量拉取，合并raw.json，构建HTML下载队列，后台异步下载（每5秒1条，下载完整才保存）
# intent: fetch只负责RSS抓取+保存raw.json+构建队列，不等待HTML下载完成；前端fetch按钮可快速返回
#   队列按source分目录保存HTML到 NEWS_STORAGE_PATH/YYYYMMDD/source/id.html
#   后台worker每5秒取一条队列项下载，下载完整后才写文件，避免部分下载的脏文件
async def handle_news_manual_fetch():
    global _fetch_running
    if _fetch_running:
        logger.warning("[News] Manual fetch requested but already running, rejected")
        return {"ok": False, "msg": "正在拉取中，请稍后再试"}
    _fetch_running = True
    logger.info("[News] Starting manual news fetch")
    try:
        clean_expired_data()
        raw_entries = await fetch_all_rss()
        date_str = datetime.now().strftime("%Y%m%d")
        added = save_raw_for_date(date_str, raw_entries)
        build_download_queue()
        await launch_download_worker()
        logger.info(f"[News] Manual fetch completed, {added} new raw entries queued for HTML download")
        return {"ok": True, "raw_count": added}
    except Exception as e:
        logger.error(f"[News] Manual fetch failed: {e}", exc_info=True)
        raise
    finally:
        _fetch_running = False
# DOC-END id=news/manual_fetch#2

# DOC-BEGIN id=news/save_raw#1 type=func v=1
# summary: 将RSS条目按日期存入raw.json，合并已有数据去重，返回新增条目数
# intent: 同一天多次fetch时先读raw.json已有id避免重复；当天首次fetch时文件不存在直接写入
#   单文件存储简化读写逻辑，前端list直接读一个文件即可
def save_raw_for_date(date_str: str, raw_entries: list) -> int:
    date_dir = f"{DATA_PATH}/{date_str}"
    os.makedirs(date_dir, exist_ok=True)
    save_path = f"{date_dir}/raw.json"
    existing = []
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existed_ids = {item["id"] for item in existing}
    new_items = []
    for item in raw_entries:
        if item["id"] not in existed_ids:
            item.setdefault("llm_score", None)
            item.setdefault("human_score", None)
            item.setdefault("html_downloaded", False)
            item.setdefault("download_failed", False)
            item.setdefault("read", False)
            item.setdefault("read_time", None)
            new_items.append(item)
    if new_items:
        existing.extend(new_items)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    return len(new_items)
# DOC-END id=news/save_raw#1

# DOC-BEGIN id=news/download_queue#1 type=func v=1
# summary: 扫描所有日期目录下的raw.json，找出HTML文件尚未下载的条目，加入全局下载队列
# intent: 每次fetch或auto_fetch时调用，保证新条目被纳入下载；已下载的自动跳过，不会重复下载
def build_download_queue():
    global _download_queue
    new_count = 0
    for date_dir in sorted(os.listdir(DATA_PATH)):
        if not date_dir.isdigit(): continue
        raw_file = f"{DATA_PATH}/{date_dir}/raw.json"
        if not os.path.exists(raw_file): continue
        with open(raw_file, "r", encoding="utf-8") as f:
            for entry in json.load(f):
                src_dir = f"{DATA_PATH}/{date_dir}/{entry['source']}"
                html_path = f"{src_dir}/{entry['id']}.html"
                if not os.path.exists(html_path) and not entry.get("download_failed", False):
                    _download_queue.append({
                        "id": entry["id"],
                        "link": entry["link"],
                        "source": entry["source"],
                        "date_str": date_dir
                    })
                    new_count += 1
    logger.info(f"[News] Download queue built, {new_count} items added (total queue: {len(_download_queue)})")
# DOC-END id=news/download_queue#1

# DOC-BEGIN id=news/download_worker#1 type=func v=1
# summary: 异步下载worker，每5秒从队列取一条，下载HTML并保存到对应source目录，下载失败跳过不影响后续
# intent: 下载完整才写文件，避免部分下载的脏数据；用aiohttp 10秒超时防卡死；
#   队列用global list简单实现，launch_download_worker保证单例运行
async def _download_worker():
    global _download_running
    _download_running = True
    try:
        while _download_queue:
            item = _download_queue.pop(0)
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(item["link"], headers={
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }) as resp:
                        if resp.status == 200:
                            html = await resp.text()
                            src_dir = f"{DATA_PATH}/{item['date_str']}/{item['source']}"
                            os.makedirs(src_dir, exist_ok=True)
                            html_path = f"{src_dir}/{item['id']}.html"
                            with open(html_path, "w", encoding="utf-8") as f:
                                f.write(html)
                            # 回写 html_downloaded 标记到 raw.json
                            raw_file = f"{DATA_PATH}/{item['date_str']}/raw.json"
                            if os.path.exists(raw_file):
                                with open(raw_file, "r", encoding="utf-8") as rf:
                                    all_entries = json.load(rf)
                                for e in all_entries:
                                    if e["id"] == item["id"]:
                                        e["html_downloaded"] = True
                                        break
                                with open(raw_file, "w", encoding="utf-8") as wf:
                                    json.dump(all_entries, wf, ensure_ascii=False, indent=2)
                            logger.info(f"[News] Downloaded HTML: {item['source']}/{item['id']}.html ({len(html)} bytes)")
                        else:
                            logger.warning(f"[News] Download failed HTTP {resp.status}: {item['link']}")
                            # 标记为下载失败，不再重试
                            raw_file = f"{DATA_PATH}/{item['date_str']}/raw.json"
                            if os.path.exists(raw_file):
                                with open(raw_file, "r", encoding="utf-8") as rf:
                                    all_entries = json.load(rf)
                                for e in all_entries:
                                    if e["id"] == item["id"]:
                                        e["download_failed"] = True
                                        break
                                with open(raw_file, "w", encoding="utf-8") as wf:
                                    json.dump(all_entries, wf, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[News] Download error: {item['link']}, error: {e}")
                # 标记为下载失败，不再重试
                raw_file = f"{DATA_PATH}/{item['date_str']}/raw.json"
                if os.path.exists(raw_file):
                    with open(raw_file, "r", encoding="utf-8") as rf:
                        all_entries = json.load(rf)
                    for e in all_entries:
                        if e["id"] == item["id"]:
                            e["download_failed"] = True
                            break
                    with open(raw_file, "w", encoding="utf-8") as wf:
                        json.dump(all_entries, wf, ensure_ascii=False, indent=2)
            if _download_queue:
                await asyncio.sleep(5)
    finally:
        _download_running = False
# DOC-END id=news/download_worker#1

# DOC-BEGIN id=news/launch_worker#1 type=func v=1
# summary: 启动下载worker，如果已有worker在运行则跳过，保证全局只有一个下载线程
# intent: 通过_download_running标志实现单例；不阻塞调用方，后台异步执行
async def launch_download_worker():
    global _download_running
    if _download_running:
        logger.debug("[News] Download worker already running, skip launch")
        return
    asyncio.create_task(_download_worker())

# DOC-BEGIN id=news/llm_score_api#1 type=func v=1
# summary: 对当天所有无打分条目批量调LLM打分，返回scored和failed计数
# intent: 作为独立API端点，前端点击"Score All"触发
# DOC-BEGIN id=news/llm_score_api#2 type=func v=2
# summary: 对当天所有无打分条目批量调LLM打分，接收model_name/api_key/llm_url参数传递给llm_score_entries
# intent: 作为独立API端点，前端点击"Score All"触发；模型配置由前端传入，支持选择不同模型进行打分
async def handle_news_llm_score(model_name: str = None, api_key: str = None, llm_url: str = None):
    date_str = datetime.now().strftime("%Y%m%d")
    scored, failed = await llm_score_entries(date_str, model_name=model_name, api_key=api_key, llm_url=llm_url)
    logger.info(f"[News] LLM scoring completed: scored={scored}, failed={failed}")
    return {"ok": True, "scored": scored, "failed": failed}
# DOC-END id=news/llm_score_api#2

# DOC-BEGIN id=news/human_score#2 type=func v=2
# summary: 接收人类打分，更新raw.json中对应条目的human_score、read=true和read_time，返回ok
# intent: 人类打分时自动标记已读并记录阅读时间，支持按阅读时间排序；按date_str+entry_id定位条目
def handle_news_score(date_str: str, entry_id: str, human_score: int):
    raw_file = f"{NEWS_STORAGE_PATH}/{date_str}/raw.json"
    if not os.path.exists(raw_file):
        return {"ok": False, "error": "Date data not found"}
    with open(raw_file, "r", encoding="utf-8") as f:
        entries = json.load(f)
    found = False
    target_entry = None
    old_score = None
    for entry in entries:
        if entry["id"] == entry_id:
            old_score = entry.get("human_score")
            entry["human_score"] = human_score
            entry["read"] = True
            entry["read_time"] = datetime.now().isoformat()
            target_entry = entry
            found = True
            break
    if not found:
        return {"ok": False, "error": "Entry not found"}
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    
    # 更新人类偏好存储
    pref = load_human_preference()
    score_key = str(human_score)
    sample = {
        "title": target_entry["title"],
        "summary": target_entry.get("summary", ""),
        "human_score": human_score
    }
    # 如果之前有旧分数，先从旧槽移除该样本（如果存在）
    if old_score is not None and str(old_score) in pref:
        old_key = str(old_score)
        pref[old_key] = [s for s in pref[old_key] if s.get("title") != target_entry["title"]]
    # 加入新槽，保持最多3个，满了踢最老的
    pref[score_key].append(sample)
    if len(pref[score_key]) > 3:
        pref[score_key] = pref[score_key][-3:]
    save_human_preference(pref)
    
    return {"ok": True}
# DOC-END id=news/human_score#2

# DOC-BEGIN id=news/serve_html#2 type=func v=2
# summary: 根据date_str和entry_id返回本地HTML文件内容，文件不存在时将该任务插入下载队列最前面优先下载，返回错误提示
# intent: 用户点击未下载新闻时触发优先下载，自动启动下载worker，确保第一时间下载；避免重复添加到队列
async def handle_news_html(date_str: str, entry_id: str):
    for date_dir in [date_str] + sorted(os.listdir(NEWS_STORAGE_PATH), reverse=True):
        if not date_dir.isdigit(): continue
        raw_file = f"{NEWS_STORAGE_PATH}/{date_dir}/raw.json"
        if not os.path.exists(raw_file): continue
        with open(raw_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            if entry["id"] == entry_id:
                html_path = f"{NEWS_STORAGE_PATH}/{date_dir}/{entry['source']}/{entry_id}.html"
                if os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as hf:
                        return {"ok": True, "html": hf.read()}
                else:
                    if entry.get("download_failed", False):
                        return {"ok": False, "error": "download_failed", "link": entry["link"]}
                    # 检查是否已在队列中，避免重复添加
                    exist_in_queue = any(item["id"] == entry_id for item in _download_queue)
                    if not exist_in_queue:
                        # 插入队列最前面，优先下载
                        _download_queue.insert(0, {
                            "id": entry_id,
                            "link": entry["link"],
                            "source": entry["source"],
                            "date_str": date_dir
                        })
                        logger.info(f"[News] Added priority download task for entry {entry_id}")
                    # 启动下载worker如果未运行
                    await launch_download_worker()
                    return {"ok": False, "error": "HTML not downloaded yet"}
    return {"ok": False, "error": "Entry not found"}
# DOC-END id=news/serve_html#2

# DOC-BEGIN id=news/auto_toggle#1 type=func v=1
# summary: 开关自动拉取功能，输入enable布尔值，返回当前状态；状态持久化到本地文件，重启不丢失
# intent: 定时任务用AsyncIOScheduler和FastAPI事件循环兼容，不需要额外进程；自动拉取间隔可配置
def handle_news_auto_toggle(enable: bool):
    global auto_fetch_enabled
    auto_fetch_enabled = enable
    with open(f"{NEWS_STORAGE_PATH}/auto_config.json", "w", encoding="utf-8") as f:
        json.dump({"enable": enable}, f)
    if enable and not scheduler.running:
        scheduler.add_job(handle_news_manual_fetch, "interval", minutes=AUTO_FETCH_INTERVAL)
        scheduler.start()
        build_download_queue()
        asyncio.create_task(launch_download_worker())
        logger.info(f"[News] Auto fetch enabled, scheduled to run every {AUTO_FETCH_INTERVAL} minutes")
    elif not enable and scheduler.running:
        scheduler.shutdown()
        logger.info("[News] Auto fetch disabled, scheduler stopped")
    logger.info(f"[News] Auto fetch status updated to: {auto_fetch_enabled}")
    return {"ok": True, "auto_fetch_enabled": auto_fetch_enabled}
# DOC-END id=news/auto_toggle#1

# DOC-BEGIN id=news/auto_status#1 type=func v=1
# summary: 查询自动拉取当前状态，返回是否启用和间隔分钟数
# intent: 前端轮询此接口同步开关状态
def handle_news_auto_status():
    return {"ok": True, "auto_fetch_enabled": auto_fetch_enabled, "interval": AUTO_FETCH_INTERVAL}
# DOC-END id=news/auto_status#1

# DOC-BEGIN id=news/list#2 type=func v=3
# summary: 分页查询新闻列表，支持筛选unread/read（unread仅返回llm_score非null且read=false，read仅返回read=true），
#   unread按llm_score降序，read按read_time降序（最近阅读优先）
# intent: unread筛选用于前端"待阅读"列表，read筛选用于"已读"列表；按阅读时间排序方便回顾最近看过的文章
# DOC-BEGIN id=news/list#3 type=func v=4
# summary: 分页查询新闻列表，支持筛选unread/read和category分类，
#   unread仅返回llm_score非null且read=false，read仅返回read=true，
#   category用于筛选特定领域（Tech/Finance/Global）
# intent: unread筛选用于前端"待阅读"列表，read筛选用于"已读"列表；
#   category筛选让用户可以按兴趣领域浏览新闻；按阅读时间排序方便回顾最近看过的文章
def handle_news_list(page: int = 1, page_size: int = 20, category: str = None, unread_only: bool = False, read_only: bool = False):
    all_news = []
    for date_dir in sorted(os.listdir(NEWS_STORAGE_PATH), reverse=True):
        if not date_dir.isdigit(): continue
        raw_file = f"{NEWS_STORAGE_PATH}/{date_dir}/raw.json"
        if not os.path.exists(raw_file): continue
        with open(raw_file, "r", encoding="utf-8") as f:
            items = json.load(f)
            if category and category != "all":
                items = [i for i in items if i.get("category") == category]
            if unread_only:
                items = [i for i in items if i.get("llm_score") is not None and not i.get("read", False)]
            elif read_only:
                items = [i for i in items if i.get("read", False)]
            for i in items:
                i["_date_str"] = date_dir
            all_news.extend(items)
    if unread_only:
        all_news.sort(key=lambda x: x.get("llm_score", 0), reverse=True)
    elif read_only:
        all_news.sort(key=lambda x: x.get("read_time", ""), reverse=True)
    total = len(all_news)
    start = (page - 1) * page_size
    end = start + page_size
    ret_list = all_news[start:end]
    logger.info(f"[News] News list query: page={page}, category={category}, unread_only={unread_only}, read_only={read_only}, total={total}, returned={len(ret_list)} entries")
    return {"ok": True, "total": total, "list": ret_list}
# DOC-END id=news/list#3

# DOC-BEGIN id=news/categories#1 type=func v=1
# summary: 获取所有可用的新闻分类列表，从RSS_SOURCES配置中提取去重后的分类
# intent: 前端需要显示分类筛选器，动态获取分类列表避免硬编码
def handle_news_categories():
    categories = sorted(set(source[0] for source in RSS_SOURCES))
    logger.info(f"[News] Available categories: {categories}")
    return {"ok": True, "categories": categories}
# DOC-END id=news/categories#1

# DOC-BEGIN id=news/scoring_status#1 type=func v=1
# summary: 查询当前LLM打分进程状态，返回是否运行、总数、已完成数
# intent: 前端轮询获取进度，刷新页面后可恢复进度显示
def handle_news_get_scoring_status():
    global _scoring_running, _total_to_score, _scored_count
    return {
        "ok": True,
        "running": _scoring_running,
        "total": _total_to_score,
        "scored": _scored_count
    }
# DOC-END id=news/scoring_status#1

# DOC-BEGIN id=news/cancel_scoring#1 type=func v=1
# summary: 取消正在运行的LLM打分进程，设置取消标志
# intent: 前端点击取消按钮触发，下一批次处理前自动终止
def handle_news_cancel_scoring():
    global _cancel_scoring_flag
    _cancel_scoring_flag = True
    logger.info("[News] Scoring cancel requested by user")
    return {"ok": True}
# DOC-END id=news/cancel_scoring#1
