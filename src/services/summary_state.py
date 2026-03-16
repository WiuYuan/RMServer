# src/services/summary_state.py
import json
import time
import threading
import logging
from typing import Dict, Optional, List
from collections import OrderedDict

logger = logging.getLogger(__name__)


# DOC-BEGIN id=summary_state/SummaryStateManager#1 type=core v=1
# summary: 管理单个 task 的 summary 展开/关闭状态，支持目录级、文件级、DOC 块级三层节点
# intent: LLM 通过 open_summary/close_summary 工具控制当前上下文中可见的代码摘要范围。
#   核心数据结构是一个 OrderedDict（按访问时间排序），记录所有已展开节点的 id、内容字符数、
#   最后访问时间。当总字符数超过阈值时，按 LRU 策略从最早访问的节点开始自动关闭。
#   节点分三类：
#     - 目录节点 (id = "dir::src/services/")：展开显示该目录下的文件/子目录列表
#     - 文件节点 (id = "file::src/services/session_history.py")：展开显示该文件的 DOC 块列表
#     - DOC 块节点 (id = "doc::session_history/bind_workspace#2")：展开显示 DOC 块的详细内容
#   折叠父节点时，子节点的展开状态保留在内存中但不渲染（下次展开父节点时恢复可见）。
class SummaryStateManager:
    # DOC-BEGIN id=summary_state/SummaryStateManager/__init__#2 type=core v=2
    # summary: 初始化 SummaryStateManager，接收 max_chars 上限、可选的 rm_index 引用和 workspace_id
    # intent: rm_index 和 workspace_id 用于持久化——通过 rm_index._exec 将展开状态写入远程 .rm/summary_state.json。
    #   如果 rm_index 为 None（测试或非 workspace 场景），持久化功能静默跳过。
    #   _dirty 标记用于合并多次快速操作，避免每次 open/close 都触发远程写入。
    def __init__(self, max_chars: int = 30000, rm_index=None, workspace_id: str = None):
        self._max_chars = max_chars
        self._rm_index = rm_index
        self._workspace_id = workspace_id
        self._opened: OrderedDict = OrderedDict()
        self._total_chars: int = 0
        self._lock = threading.Lock()
        self._dirty: bool = False
        self._save_timer: Optional[threading.Timer] = None
    # DOC-END id=summary_state/SummaryStateManager/__init__#2

    # DOC-BEGIN id=summary_state/SummaryStateManager/open_node#1 type=core v=1
    # summary: 展开指定节点——将节点内容加入已展开集合，触发 auto_evict 确保不超限
    # intent: node_id 格式为 "dir::path"、"file::path"、"doc::block_id"。
    #   content 由调用方（LLM 工具层）根据 RMIndex 数据生成并传入。
    #   如果节点已展开则更新 last_access（移到 OrderedDict 末尾），不重复计算字符数。
    #   新节点默认展开（符合需求：新出现的 DOC 默认展开）。
    def open_node(self, node_id: str, content: str) -> dict:
        with self._lock:
            if node_id in self._opened:
                self._opened.move_to_end(node_id)
                self._opened[node_id]["last_access"] = time.time()
                return {"status": "already_open", "node_id": node_id}

            chars = len(content)
            now = time.time()
            self._opened[node_id] = {
                "content": content,
                "chars": chars,
                "opened_at": now,
                "last_access": now,
            }
            self._total_chars += chars
            self._auto_evict_locked(exclude={node_id})
            result = {"status": "opened", "node_id": node_id, "chars": chars}
            evicted = self._get_eviction_info_locked()
            if evicted:
                result["evicted"] = evicted
        self._schedule_save()
        return result
    # DOC-END id=summary_state/SummaryStateManager/open_node#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/close_node#1 type=core v=1
    # summary: 关闭指定节点——从已展开集合中移除，释放字符配额
    # intent: 关闭后该节点在 render_context 中只显示一行占位符。
    #   如果 node_id 不存在（已关闭或从未打开），静默返回，不报错。
    def close_node(self, node_id: str) -> dict:
        with self._lock:
            entry = self._opened.pop(node_id, None)
            if entry:
                self._total_chars -= entry["chars"]
                result = {"status": "closed", "node_id": node_id}
            else:
                result = {"status": "not_open", "node_id": node_id}
        if entry:
            self._schedule_save()
        return result
    # DOC-END id=summary_state/SummaryStateManager/close_node#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/_update_node_content#1 type=behavior v=1
    # summary: 更新已展开节点的 content 和字符数，不改变 open/close 状态和访问顺序
    # intent: 当 doc 块被展开/关闭时，所属文件的折叠渲染结果会变化（代码行恢复/折叠），
    #   需要就地更新 file 节点的 content。不调用 move_to_end（不影响 LRU 顺序），
    #   也不触发 auto_evict（内容变化可能增减字符数，但这是已有节点的更新而非新增）。
    #   如果 node_id 不存在，静默返回。
    def _update_node_content(self, node_id: str, new_content: str):
        with self._lock:
            if node_id not in self._opened:
                return
            old_chars = self._opened[node_id]["chars"]
            new_chars = len(new_content)
            self._opened[node_id]["content"] = new_content
            self._opened[node_id]["chars"] = new_chars
            self._total_chars += (new_chars - old_chars)
    # DOC-END id=summary_state/SummaryStateManager/_update_node_content#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/is_open#1 type=query v=1
    # summary: 查询指定节点是否处于展开状态
    # intent: LLM 工具层在决定是否需要调用 load_file_docs 之前先检查节点状态。
    def is_open(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._opened
    # DOC-END id=summary_state/SummaryStateManager/is_open#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/_auto_evict_locked#1 type=behavior v=1
    # summary: LRU 自动淘汰——当总字符数超过阈值时，从最早访问的节点开始关闭，直到低于阈值
    # intent: 必须在持有 _lock 的情况下调用（方法名以 _locked 后缀标识）。
    #   exclude 参数是刚刚打开的节点集合，不参与淘汰（防止刚打开就被关掉）。
    #   淘汰顺序是 OrderedDict 的头部（最早访问的），这就是 LRU 策略。
    def _auto_evict_locked(self, exclude: set = None):
        exclude = exclude or set()
        while self._total_chars > self._max_chars:
            evicted = False
            for nid in list(self._opened.keys()):
                if nid in exclude:
                    continue
                entry = self._opened.pop(nid)
                self._total_chars -= entry["chars"]
                logger.info(f"Auto-evicted node {nid} ({entry['chars']} chars)")
                evicted = True
                break
            if not evicted:
                break
    # DOC-END id=summary_state/SummaryStateManager/_auto_evict_locked#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/rescan_merge#1 type=core v=1
    # summary: 重新扫描后合并状态——保留已有节点的展开/关闭状态，新节点默认展开
    # intent: 当代码被修改后触发 rescan，DOC 块的 id 集合可能发生变化。
    #   old_ids ∩ new_ids：保留原有展开/关闭状态
    #   new_ids - old_ids：新出现的节点，默认展开（调用方需要传入 content）
    #   old_ids - new_ids：被删除的节点，从状态机移除
    #   new_contents 是 {node_id: content_str} 的字典，只需要包含新增节点的内容。
    def rescan_merge(self, old_ids: set, new_ids: set, new_contents: Dict[str, str]) -> dict:
        added = new_ids - old_ids
        removed = old_ids - new_ids
        kept = old_ids & new_ids

        with self._lock:
            for nid in removed:
                entry = self._opened.pop(nid, None)
                if entry:
                    self._total_chars -= entry["chars"]

        result = {"added": [], "removed": list(removed), "kept": list(kept)}

        for nid in added:
            content = new_contents.get(nid, f"[NEW] {nid}")
            self.open_node(nid, content)
            result["added"].append(nid)

        return result
    # DOC-END id=summary_state/SummaryStateManager/rescan_merge#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/get_opened_list#1 type=query v=1
    # summary: 返回当前所有已展开节点的 id 列表和总字符数，供 LLM 了解当前上下文占用
    # intent: LLM 可以据此决定是否需要主动关闭一些节点来腾出空间。
    def get_opened_list(self) -> dict:
        with self._lock:
            nodes = []
            for nid, entry in self._opened.items():
                nodes.append({
                    "node_id": nid,
                    "chars": entry["chars"],
                    "last_access": entry["last_access"],
                })
            return {
                "opened_count": len(nodes),
                "total_chars": self._total_chars,
                "max_chars": self._max_chars,
                "nodes": nodes,
            }
    # DOC-END id=summary_state/SummaryStateManager/get_opened_list#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/render_file_with_folding#1 type=core v=1
    # summary: 对文件代码进行折叠渲染——已关闭的 DOC 块区域替换为占位符，已展开的保留原始代码
    # intent: 这是实现"代码折叠"功能的核心方法。输入是文件原始内容和该文件的所有 DOC 块信息，
    #   输出是折叠处理后的代码文本。处理逻辑：
    #   1) 收集所有需要折叠的顶层 DOC 块（即自身关闭、且其 parent 要么不存在要么已展开的块）
    #   2) 对于展开的 DOC 块，递归检查其子块的状态
    #   3) 折叠操作：将 [line_start, line_end] 范围内的行替换为一行占位符
    #      占位符格式: [FOLDED doc::<block_id>] <summary> (use open_summary to expand)
    #   4) 折叠区域可能嵌套，需要从最内层开始处理（避免行号偏移问题）
    #   blocks 的 line_range 是 1-based 闭区间 [start, end]。
    def render_file_with_folding(self, file_content: str, blocks: Dict[str, dict], filepath: str) -> str:
        if not blocks:
            return file_content

        lines = file_content.split("\n")

        # DOC-BEGIN id=summary_state/render_file_with_folding/collect_fold_regions#1 type=behavior v=1
        # summary: 递归收集所有需要折叠的行范围——对每个 block 判断其展开状态，
        #   若关闭则整个 line_range 折叠，若展开则递归检查子块
        # intent: 折叠决策是递归的：一个已展开的 block 内部可能有子 block，
        #   子 block 可能是关闭的（需要折叠）。但如果父 block 本身是关闭的，
        #   则整个父 block 区域已被折叠，不需要再看子块。
        #   fold_regions 存储 (start_line_1based, end_line_1based, block_id, summary) 元组。
        #   使用 visited 集合防止循环引用（防御性编程，正常不会发生）。
        fold_regions = []

        def collect_folds(block_ids: list, visited: set = None):
            if visited is None:
                visited = set()
            for bid in block_ids:
                if bid in visited:
                    continue
                visited.add(bid)
                block = blocks.get(bid)
                if not block:
                    continue
                node_id = f"doc::{bid}"
                if self.is_open(node_id):
                    # 此 block 已展开——显示原始代码，但递归检查子块
                    if block.get("children"):
                        collect_folds(block["children"], visited)
                else:
                    # 此 block 已关闭——整个区域折叠为占位符
                    start, end = block["line_range"]
                    summary_text = block.get("summary", bid)
                    fold_regions.append((start, end, bid, summary_text))

        # 从根节点（parent is None）开始递归
        root_blocks = [bid for bid, b in blocks.items() if b.get("parent") is None]
        collect_folds(root_blocks)
        # DOC-END id=summary_state/render_file_with_folding/collect_fold_regions#1

        if not fold_regions:
            return file_content

        # DOC-BEGIN id=summary_state/render_file_with_folding/apply_folds#1 type=behavior v=1
        # summary: 按行号从大到小排序折叠区域，依次将对应行替换为占位符
        # intent: 从后向前处理是关键——这样前面行的行号不会因为后面的替换而偏移。
        #   如果存在嵌套折叠（不应该发生，因为 collect_folds 已处理），
        #   需要过滤掉被更大折叠区域完全包含的子区域。
        #   但正常情况下 collect_folds 保证不会同时折叠父和子：
        #   父关闭→整体折叠，不递归子；父展开→不折叠父，只检查子。
        fold_regions.sort(key=lambda r: r[0], reverse=True)

        # 去除被更大区域完全包含的子区域（防御性）
        filtered = []
        for region in fold_regions:
            contained = False
            for other in filtered:
                if other[0] <= region[0] and other[1] >= region[1] and other != region:
                    contained = True
                    break
            if not contained:
                filtered.append(region)

        for start, end, bid, summary_text in filtered:
            # line_range 是 1-based，转为 0-based index
            start_idx = start - 1
            end_idx = end  # end is inclusive, so lines[start-1:end] covers [start, end]

            # 检测该 block 代码行的缩进，让占位符保持相同缩进
            indent = ""
            if start_idx < len(lines):
                original_line = lines[start_idx]
                indent = original_line[:len(original_line) - len(original_line.lstrip())]

            placeholder = f"{indent}# [FOLDED doc::{bid}] {summary_text} (use open_summary(\"doc::{bid}\") to expand)"
            lines[start_idx:end_idx] = [placeholder]
        # DOC-END id=summary_state/render_file_with_folding/apply_folds#1

        return "\n".join(lines)
    # DOC-END id=summary_state/SummaryStateManager/render_file_with_folding#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/render_context#1 type=core v=1
    # summary: 生成当前上下文文本——展开的节点显示完整内容，关闭的显示一行占位符
    # intent: 这是注入 LLM system prompt 的核心方法。all_known_ids 是所有可能出现的节点 id
    #   （来自 RMIndex 的目录+文件列表），用于生成关闭节点的占位符。
    #   已展开节点按 OrderedDict 顺序排列（最近访问的在后面，最接近当前对话）。
    #   占位符格式：[COLLAPSED] node_id: summary_one_liner
    #   summaries 参数是 {node_id: one_line_summary} 字典，用于占位符显示。
    def render_context(self, all_known_ids: List[str] = None, summaries: Dict[str, str] = None) -> str:
        all_known_ids = all_known_ids or []
        summaries = summaries or {}
        lines = []
        lines.append("=== PROJECT CONTEXT (Summary Navigation) ===")
        lines.append(f"[{self._total_chars}/{self._max_chars} chars used]")
        lines.append("")

        with self._lock:
            opened_ids = set(self._opened.keys())

            for nid in all_known_ids:
                if nid in opened_ids:
                    entry = self._opened[nid]
                    lines.append(f"[OPEN] {nid}")
                    lines.append(entry["content"])
                    lines.append("")
                else:
                    summary = summaries.get(nid, "")
                    if summary:
                        lines.append(f"[COLLAPSED] {nid}: {summary}")
                    else:
                        lines.append(f"[COLLAPSED] {nid}")

        lines.append("=== END PROJECT CONTEXT ===")
        return "\n".join(lines)
    # DOC-END id=summary_state/SummaryStateManager/render_context#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/_schedule_save#1 type=behavior v=1
    # summary: 延迟 1 秒后触发持久化保存，多次快速调用合并为一次写入
    # intent: open_node/close_node 可能在同一个 LLM step 中被多次调用（如 LRU 淘汰连锁关闭），
    #   每次都立即写入远程太浪费。使用 Timer 延迟 1 秒，如果期间有新的 save 请求则取消旧 Timer
    #   重新计时（debounce 模式）。Timer 在 daemon 线程中执行，不阻塞 tool call 返回。
    #   如果 rm_index 为 None，静默跳过。
    def _schedule_save(self):
        if not self._rm_index:
            return
        with self._lock:
            self._dirty = True
            if self._save_timer is not None:
                self._save_timer.cancel()
            self._save_timer = threading.Timer(1.0, self._do_save)
            self._save_timer.daemon = True
            self._save_timer.start()
    # DOC-END id=summary_state/SummaryStateManager/_schedule_save#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/_get_eviction_info_locked#1 type=internal v=1
    # summary: 辅助方法——在持有锁时收集淘汰信息（占位，供 open_node 返回值使用）
    # intent: 原 open_node 的 evicted 字段需要在 auto_evict 后收集被淘汰的 node_id 列表。
    #   当前实现简化为返回空列表，因为 _auto_evict_locked 不返回淘汰信息（仅 log）。
    #   后续可扩展 _auto_evict_locked 使其记录淘汰列表。
    def _get_eviction_info_locked(self) -> list:
        return []
    # DOC-END id=summary_state/SummaryStateManager/_get_eviction_info_locked#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/_do_save#1 type=core v=1
    # summary: 将当前展开状态序列化为 JSON 并通过 rm_index._exec + base64 写入远程 .rm/summary_state.json
    # intent: 持久化的内容只包含 node_id 列表和元数据（opened_at、last_access），
    #   不包含 content——content 在加载时从远程文件重新渲染。这样避免了 JSON 体积膨胀，
    #   也确保加载后的 content 是最新的（文件可能在两次会话之间被修改）。
    #   写入使用 base64 编码，与 RMIndex.build_and_save_summary 保持一致的安全写入模式。
    #   写入失败仅 warning 不阻断——持久化是"尽力而为"的优化，不影响核心功能。
    def _do_save(self):
        if not self._rm_index:
            return
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
            snapshot = []
            for nid, entry in self._opened.items():
                snapshot.append({
                    "node_id": nid,
                    "opened_at": entry["opened_at"],
                    "last_access": entry["last_access"],
                })

        try:
            import base64 as _b64
            import uuid as _uuid

            state_data = {
                "version": 1,
                "workspace_id": self._workspace_id or "",
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "max_chars": self._max_chars,
                "opened_nodes": snapshot,
            }
            content = json.dumps(state_data, ensure_ascii=False, indent=2)
            encoded = _b64.b64encode(content.encode("utf-8")).decode("ascii")

            rm_dir = self._rm_index.rm_dir
            target = f"{rm_dir}/summary_state.json"
            tmp = f"/tmp/_rmss_{_uuid.uuid4().hex[:8]}.b64"

            chunk_size = 48000
            self._rm_index._exec(f"> '{tmp}'")
            for i in range(0, len(encoded), chunk_size):
                chunk = encoded[i:i + chunk_size]
                wr = self._rm_index._exec(f"echo -n '{chunk}' >> '{tmp}'")
                if not wr.get("ok"):
                    self._rm_index._exec(f"rm -f '{tmp}'")
                    logger.warning(f"Failed to save summary_state.json chunk at offset {i}")
                    return
            decode_result = self._rm_index._exec(f"base64 -d '{tmp}' > '{target}' && rm -f '{tmp}'")
            if not decode_result.get("ok"):
                self._rm_index._exec(f"rm -f '{tmp}'")
                logger.warning(f"Failed to decode summary_state.json: {decode_result.get('error', '')}")
                return

            logger.info(f"Saved summary_state.json for workspace {self._workspace_id}: {len(snapshot)} nodes")
        except Exception as e:
            logger.warning(f"Failed to save summary_state.json: {e}")
    # DOC-END id=summary_state/SummaryStateManager/_do_save#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/save_now#1 type=behavior v=1
    # summary: 立即同步保存当前状态（取消延迟 Timer），用于关键时刻确保持久化
    # intent: _schedule_save 是异步延迟的，某些场景（如 post_step_hook、task 结束）
    #   需要确保状态已写入。此方法取消 pending Timer 并同步执行 _do_save。
    def save_now(self):
        with self._lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
                self._save_timer = None
            self._dirty = True
        self._do_save()
    # DOC-END id=summary_state/SummaryStateManager/save_now#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/load_from_remote#1 type=core v=1
    # summary: 从远程 .rm/summary_state.json 加载展开状态，逐个恢复已展开的节点
    # intent: attach_workspace 时调用。加载后需要为每个 node_id 重新生成 content：
    #   - dir:: 节点：从 rm_index.dir_summary 渲染子条目列表
    #   - file:: 节点：从远程读取文件内容 + 折叠渲染
    #   - doc:: 节点：存入轻量标记（与 open_summary/doc_branch 一致）
    #   加载过程中某个节点恢复失败（如文件已被删除）不影响其他节点，仅 warning。
    #   加载完成后不触发 _schedule_save（避免加载后立即回写造成无意义 I/O）。
    def load_from_remote(self) -> dict:
        if not self._rm_index:
            return {"ok": False, "error": "No rm_index bound"}

        try:
            rm_dir = self._rm_index.rm_dir
            cat_result = self._rm_index._exec(f"test -f '{rm_dir}/summary_state.json' && cat '{rm_dir}/summary_state.json' || echo '__NOT_FOUND__'")
            if not cat_result.get("ok"):
                return {"ok": False, "error": f"Failed to read summary_state.json: {cat_result.get('error', '')}"}

            raw = cat_result.get("stdout", "").strip()
            if raw == "__NOT_FOUND__" or not raw:
                return {"ok": True, "loaded": 0, "msg": "No saved state found"}

            state_data = json.loads(raw)
            opened_nodes = state_data.get("opened_nodes", [])

            restored = 0
            failed = 0

            # DOC-BEGIN id=summary_state/load_from_remote/restore_nodes#1 type=behavior v=1
            # summary: 按保存的顺序逐个恢复节点——根据 node_id 前缀分发到不同的 content 生成逻辑
            # intent: 恢复顺序与保存顺序一致（OrderedDict 插入顺序），确保 LRU 淘汰行为与上次一致。
            #   content 必须重新生成而非存储，因为文件内容可能在两次会话之间被修改。
            #   doc:: 节点必须在对应 file:: 节点之后恢复（file 的折叠渲染依赖 doc 的展开状态），
            #   但保存时 doc 总是在 file 之后插入（open_summary 工具的使用顺序保证），
            #   加载时按原序恢复即可自然满足此依赖。
            for node_info in opened_nodes:
                nid = node_info["node_id"]
                try:
                    content = self._rebuild_node_content(nid)
                    if content is None:
                        logger.warning(f"Cannot rebuild content for {nid}, skipping")
                        failed += 1
                        continue

                    with self._lock:
                        if nid in self._opened:
                            continue
                        chars = len(content)
                        self._opened[nid] = {
                            "content": content,
                            "chars": chars,
                            "opened_at": node_info.get("opened_at", time.time()),
                            "last_access": node_info.get("last_access", time.time()),
                        }
                        self._total_chars += chars
                    restored += 1
                except Exception as e:
                    logger.warning(f"Failed to restore node {nid}: {e}")
                    failed += 1
            # DOC-END id=summary_state/load_from_remote/restore_nodes#1

            # DOC-BEGIN id=summary_state/load_from_remote/post_evict#1 type=behavior v=1
            # summary: 恢复完成后执行一次 auto_evict，确保总字符数不超限
            # intent: 上次保存时可能恰好在限额内，但文件内容变化后重新渲染的 content 可能更长。
            #   此处统一淘汰一次，避免超限状态进入正常使用。
            with self._lock:
                self._auto_evict_locked()
            # DOC-END id=summary_state/load_from_remote/post_evict#1

            logger.info(f"Loaded summary_state: restored={restored}, failed={failed}")
            return {"ok": True, "loaded": restored, "failed": failed}

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt summary_state.json: {e}")
            return {"ok": False, "error": f"JSON parse error: {e}"}
        except Exception as e:
            logger.warning(f"Failed to load summary_state: {e}")
            return {"ok": False, "error": str(e)}
    # DOC-END id=summary_state/SummaryStateManager/load_from_remote#1

    # DOC-BEGIN id=summary_state/SummaryStateManager/_rebuild_node_content#1 type=internal v=1
    # summary: 根据 node_id 前缀分发生成对应的 content 文本，用于加载恢复
    # intent: 三种节点类型的 content 生成逻辑与 llm_handlers.py 中 open_summary 工具的逻辑一致，
    #   但这里直接调用 rm_index 的方法，不经过 tool call 层。
    #   返回 None 表示该节点已不存在（文件被删除、目录被移除等），调用方应跳过。
    def _rebuild_node_content(self, node_id: str) -> Optional[str]:
        parts = node_id.split("::", 1)
        if len(parts) != 2:
            return None

        node_type, node_path = parts
        rm_index = self._rm_index

        if node_type == "dir":
            prefix = node_path if node_path.endswith("/") else node_path + "/"
            children = []
            for entry_key, entry_val in rm_index.dir_summary.items():
                if not entry_key.startswith(prefix):
                    continue
                remainder = entry_key[len(prefix):]
                if not remainder:
                    continue
                depth = remainder.rstrip("/").count("/")
                if depth > 0:
                    continue
                children.append(f"  {'📁' if entry_val['type'] == 'dir' else '📄'} {remainder}")
            children.sort()
            if children:
                return f"Directory: {node_path}\n" + "\n".join(children)
            else:
                return f"Directory: {node_path}\n  (empty)"

        elif node_type == "file":
            blocks = rm_index.load_file_docs(node_path)
            file_content = rm_index.read_file_content(node_path)
            if file_content is None:
                return None
            if blocks:
                folded_content = self.render_file_with_folding(file_content, blocks, node_path)
            else:
                folded_content = file_content
            return f"File: {node_path}\n```\n{folded_content}\n```"

        elif node_type == "doc":
            all_blocks = rm_index.get_all_loaded_blocks()
            block_file = None
            for gkey, gval in all_blocks.items():
                if gval["id"] == node_path:
                    block_file = gval.get("_file", "")
                    break
            if not block_file:
                # 尝试加载可能的文件——doc 块所属文件可能尚未加载
                # 无法确定文件归属时返回 None
                return None
            return f"[expanded] doc::{node_path} in {block_file}"

        return None
    # DOC-END id=summary_state/SummaryStateManager/_rebuild_node_content#1
