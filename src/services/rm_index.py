
# src/services/rm_index.py
import re
import json
import time
import hashlib
import threading
import logging
import uuid
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# DOC-BEGIN id=rm_index/doc_block_schema#1 type=design v=1
# summary: DocBlock 数据类，存储从源码中解析出的单个 DOC-BEGIN/END 块的全部信息
# intent: 用 dict 而非 dataclass，因为需要频繁序列化为 JSON 存入 .rm/docs/。
#   parent/children 通过 line_range 包含关系计算，不依赖 id 命名约定。
#   line_range 是 [start, end] 闭区间，指 DOC-BEGIN 所在行到 DOC-END 所在行。
# DOC-END id=rm_index/doc_block_schema#1

# DOC-BEGIN id=rm_index/parse_doc_blocks#1 type=core v=1
# summary: 从文件内容字符串中提取所有 DOC-BEGIN/END 块，返回 {id: block_dict} 的字典
# intent: 使用正则逐行扫描，维护一个栈来处理嵌套。遇到 DOC-BEGIN 压栈，
#   遇到 DOC-END 出栈并匹配 id。如果 id 不匹配则记录警告并跳过（容错）。
#   支持任意注释前缀（#、//、--、;），通过 lstrip 后匹配 DOC-BEGIN/END 关键字。
#   fields 只提取 summary 和 intent（后续可扩展），多行 field 值目前不支持。
def parse_doc_blocks(content: str) -> Dict[str, dict]:
    lines = content.split("\n")
    blocks = {}
    stack = []

    re_begin = re.compile(
        r'^[#/;*\-\s]*DOC-BEGIN\s+id=(\S+)'
    )
    re_end = re.compile(
        r'^[#/;*\-\s]*DOC-END\s+id=(\S+)'
    )
    re_field = re.compile(
        r'^[#/;*\-\s]*(summary|intent|type|v):\s*(.*)'
    )

    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()

        m_begin = re_begin.match(stripped)
        if m_begin:
            block_id = m_begin.group(1)
            stack.append({
                "id": block_id,
                "line_start": line_num,
                "fields": {},
            })
            continue

        m_end = re_end.match(stripped)
        if m_end:
            end_id = m_end.group(1)
            if not stack:
                logger.warning(f"DOC-END id={end_id} at line {line_num} without matching BEGIN, skipped")
                continue
            top = stack[-1]
            if top["id"] != end_id:
                logger.warning(
                    f"DOC-END id={end_id} at line {line_num} does not match "
                    f"top of stack id={top['id']} (line {top['line_start']}), skipped"
                )
                continue
            stack.pop()
            blocks[end_id] = {
                "id": end_id,
                "line_range": [top["line_start"], line_num],
                "summary": top["fields"].get("summary", ""),
                "intent": top["fields"].get("intent", ""),
                "parent": None,
                "children": [],
            }
            continue

        if stack:
            m_field = re_field.match(stripped)
            if m_field:
                field_name = m_field.group(1)
                field_value = m_field.group(2).strip()
                if field_name in ("summary", "intent"):
                    stack[-1]["fields"][field_name] = field_value

    if stack:
        for orphan in stack:
            logger.warning(
                f"DOC-BEGIN id={orphan['id']} at line {orphan['line_start']} "
                f"has no matching DOC-END, discarded"
            )

    return blocks
# DOC-END id=rm_index/parse_doc_blocks#1


# DOC-BEGIN id=rm_index/build_tree#1 type=core v=1
# summary: 根据 line_range 包含关系，为所有 blocks 计算 parent 和 children 字段
# intent: 嵌套判定的唯一可靠依据是源码物理位置——如果 A 的 line_range 完全包含 B 的 line_range，
#   则 B 是 A 的后代。直接父节点是"包含 B 且 line_range 最小的那个 block"。
#   算法：将所有 block 按 line_range 长度从大到小排序，逐个插入，找到最紧包围的已有 block 作为 parent。
#   时间复杂度 O(n^2)，对于单文件的 DOC 块数量（通常 <100）完全够用。
def build_tree(blocks: Dict[str, dict]) -> Dict[str, dict]:
    block_list = sorted(blocks.values(), key=lambda b: b["line_range"][0])

    for b in block_list:
        b["parent"] = None
        b["children"] = []

    for i, child in enumerate(block_list):
        best_parent = None
        best_span = float("inf")
        c_start, c_end = child["line_range"]

        for j, candidate in enumerate(block_list):
            if i == j:
                continue
            p_start, p_end = candidate["line_range"]
            if p_start < c_start and p_end > c_end:
                span = p_end - p_start
                if span < best_span:
                    best_span = span
                    best_parent = candidate

        if best_parent:
            child["parent"] = best_parent["id"]
            best_parent["children"].append(child["id"])

    result = {}
    for b in block_list:
        result[b["id"]] = b
    return result
# DOC-END id=rm_index/build_tree#1


# DOC-BEGIN id=rm_index/scan_file_docs#1 type=core v=1
# summary: 解析单个文件内容的 DOC 块并构建嵌套树，返回完整的 blocks 字典
# intent: 组合 parse_doc_blocks + build_tree 的便捷方法。
#   调用方传入文件内容字符串即可，不涉及远程 I/O（I/O 由上层负责）。
def scan_file_docs(content: str) -> Dict[str, dict]:
    blocks = parse_doc_blocks(content)
    if blocks:
        blocks = build_tree(blocks)
    return blocks
# DOC-END id=rm_index/scan_file_docs#1


# DOC-BEGIN id=rm_index/RMIndex#1 type=core v=1
# summary: .rm/ 索引管理器，负责远程 .rm/ 目录的创建、读写、扫描、以及 DOC 块的按需加载
# intent: 一个 RMIndex 实例绑定到一个 task_id，通过 session_history 的远程文件操作
#   （exec_command_reliable / remote_read_file / remote_write_file）与远程服务器交互。
#   设计为惰性加载：bind 时只扫描目录结构建 summary.json，
#   DOC 块在 open_file_docs(filepath) 时才从远程读取并解析。
class RMIndex:
    # DOC-BEGIN id=rm_index/RMIndex/__init__#2 type=core v=2
    # summary: RMIndex 初始化——接收 workspace_id、work_dir、history_mgr、bound_terminal_id，
    #   建立与远程服务器交互的通道
    # intent: 新架构下 RMIndex 绑定到 workspace（而非 task）。bound_terminal_id 是可选参数：
    #   - workspace 创建时直接传入（此时 task→workspace 关联尚未建立，不能通过 task_id 查找）
    #   - 之后 _exec 优先使用 bound_terminal_id 直接执行命令
    #   workspace_id 用于标识日志和缓存键，不用于查找 terminal。
    def __init__(self, workspace_id: str = None, work_dir: str = None, history_mgr=None,
                 bound_terminal_id: str = None, task_id: str = None):
        self.workspace_id = workspace_id
        self.task_id = task_id
        self.work_dir = work_dir
        self._history = history_mgr
        self._bound_terminal_id = bound_terminal_id
        self.rm_dir = f"{work_dir}/.rm"

        self.dir_summary: Dict[str, dict] = {}
        self.file_docs: Dict[str, Dict[str, dict]] = {}

        self._lock = threading.Lock()
    # DOC-END id=rm_index/RMIndex/__init__#2

    # DOC-BEGIN id=rm_index/RMIndex/_exec#2 type=internal v=2
    # summary: 封装远程命令执行——优先使用 bound_terminal_id 直接操作，回退到 task_id 查找链
    # intent: workspace 创建阶段 task_id 可能为 None（workspace 尚未关联到 task），
    #   此时必须通过 bound_terminal_id 直接执行。正常运行时两种路径都可用，
    #   bound_terminal_id 路径更高效（跳过 task→workspace→terminal 的间接查找）。
    def _exec(self, command: str, timeout: float = 30.0) -> dict:
        if self._bound_terminal_id:
            return self._history.exec_command_reliable(
                task_id=self.task_id or "",
                command=command,
                timeout=timeout,
                bound_terminal_id=self._bound_terminal_id,
            )
        return self._history.exec_command_reliable(
            task_id=self.task_id or "",
            command=command,
            timeout=timeout,
        )
    # DOC-END id=rm_index/RMIndex/_exec#2

    # DOC-BEGIN id=rm_index/RMIndex/check_rm_exists#1 type=behavior v=1
    # summary: 检查远程 .rm/ 目录和 summary.json 是否存在
    # intent: bind 流程第一步：如果 .rm/summary.json 已存在，直接加载；否则需要全量扫描。
    #   返回 (rm_dir_exists, summary_json_exists) 的元组。
    def check_rm_exists(self) -> Tuple[bool, bool]:
        dir_result = self._exec(f"test -d '{self.rm_dir}' && echo 'YES' || echo 'NO'")
        dir_exists = dir_result.get("ok") and "YES" in dir_result.get("stdout", "")

        summary_result = self._exec(
            f"test -f '{self.rm_dir}/summary.json' && echo 'YES' || echo 'NO'"
        )
        summary_exists = summary_result.get("ok") and "YES" in summary_result.get("stdout", "")

        return dir_exists, summary_exists
    # DOC-END id=rm_index/RMIndex/check_rm_exists#1

    # DOC-BEGIN id=rm_index/RMIndex/scan_directory_structure#1 type=core v=1
    # summary: 扫描工作区目录结构，生成文件/目录级别的 summary 索引
    # intent: 使用 find 命令获取所有文件和目录（排除 .rm/、.git/、node_modules/、__pycache__/ 等），
    #   按 type=f 和 type=d 分别记录。此时不读取文件内容，不解析 DOC 块。
    #   summary 和 intent 字段留空，后续由 LLM 或人工填充。
    #   每个目录生成独立的 entries 段，整体存储为扁平的 {relative_path: entry} 结构。
    def scan_directory_structure(self) -> dict:
        exclude_dirs = [".rm", ".git", "node_modules", "__pycache__", ".venv", "venv", ".next"]
        prune_expr = " -o ".join([f"-name '{d}'" for d in exclude_dirs])

        # DOC-BEGIN id=rm_index/RMIndex/scan_directory_structure/find_dirs#1 type=behavior v=1
        # summary: 用 find -type d 一次性获取所有目录路径，放入 set 用于后续判断
        # intent: 原来逐个文件执行 test -d 导致 N 次远程命令调用，500 个文件就要 500 次。
        #   改为两次 find（一次 -type d，一次 -type f）即可覆盖所有条目，总共只需 2 次远程命令。
        dir_cmd = (
            f"cd '{self.work_dir}' && "
            f"find . \\( {prune_expr} \\) -prune -o -type d -print | sort"
        )
        dir_result = self._exec(dir_cmd, timeout=60.0)
        if not dir_result["ok"]:
            raise RuntimeError(f"Failed to scan directories: {dir_result.get('error', 'unknown')}")

        dir_set = set()
        for line in dir_result["stdout"].strip().split("\n"):
            line = line.strip()
            if not line or line == ".":
                continue
            if line.startswith("./"):
                line = line[2:]
            dir_set.add(line)
        # DOC-END id=rm_index/RMIndex/scan_directory_structure/find_dirs#1

        # DOC-BEGIN id=rm_index/RMIndex/scan_directory_structure/find_files#1 type=behavior v=1
        # summary: 用 find -type f 一次性获取所有文件路径
        # intent: 与 find_dirs 配合，两次 find 命令完成整个目录树的扫描，避免逐文件 test -d。
        file_cmd = (
            f"cd '{self.work_dir}' && "
            f"find . \\( {prune_expr} \\) -prune -o -type f -print | sort"
        )
        file_result = self._exec(file_cmd, timeout=60.0)
        if not file_result["ok"]:
            raise RuntimeError(f"Failed to scan files: {file_result.get('error', 'unknown')}")

        file_set = set()
        for line in file_result["stdout"].strip().split("\n"):
            line = line.strip()
            if not line or line == ".":
                continue
            if line.startswith("./"):
                line = line[2:]
            file_set.add(line)
        # DOC-END id=rm_index/RMIndex/scan_directory_structure/find_files#1

        entries = {}
        for d in sorted(dir_set):
            entries[d + "/"] = {"type": "dir", "summary": "", "intent": ""}
        for f in sorted(file_set):
            entries[f] = {"type": "file", "summary": "", "intent": ""}

        return entries
    # DOC-END id=rm_index/RMIndex/scan_directory_structure#1

    # DOC-BEGIN id=rm_index/RMIndex/build_and_save_summary#1 type=core v=1
    # summary: 全量扫描目录结构并将结果写入远程 .rm/summary.json
    # intent: bind_workspace 时如果 .rm/ 不存在，调用此方法进行初始化扫描。
    #   先创建 .rm/ 和 .rm/docs/ 目录，再扫描目录结构，最后写入 summary.json。
    #   写入使用 history_manager.remote_write_file 确保内容完整。
    def build_and_save_summary(self) -> dict:
        self._exec(f"mkdir -p '{self.rm_dir}/docs'")

        entries = self.scan_directory_structure()

        summary_data = {
            "version": 1,
            "work_dir": self.work_dir,
            "scan_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entries": entries,
        }

        # DOC-BEGIN id=rm_index/RMIndex/build_and_save_summary/write_via_exec#1 type=behavior v=1
        # summary: 通过 _exec + base64 编码安全写入 summary.json 到远程 .rm/ 目录
        # intent: 不再依赖 remote_write_file（需要 task_id），改为直接通过 bound_terminal 的 _exec 写入。
        #   使用 base64 编码确保 JSON 内容中的特殊字符不被 shell 解释。
        #   分段传输防止超过 ARG_MAX 限制。
        import base64 as _b64
        content = json.dumps(summary_data, ensure_ascii=False, indent=2)
        _encoded = _b64.b64encode(content.encode("utf-8")).decode("ascii")
        _tmp = f"/tmp/_rmsum_{uuid.uuid4().hex[:8]}.b64"
        _target = f"{self.rm_dir}/summary.json"
        # 分段写入 base64
        _chunk_size = 48000
        self._exec(f"> '{_tmp}'")
        for _i in range(0, len(_encoded), _chunk_size):
            _chunk = _encoded[_i:_i + _chunk_size]
            _wr = self._exec(f"echo -n '{_chunk}' >> '{_tmp}'")
            if not _wr.get("ok"):
                self._exec(f"rm -f '{_tmp}'")
                raise RuntimeError(f"Failed writing summary.json chunk at offset {_i}")
        _decode = self._exec(f"base64 -d '{_tmp}' > '{_target}' && rm -f '{_tmp}'")
        if not _decode.get("ok"):
            self._exec(f"rm -f '{_tmp}'")
            raise RuntimeError(f"Failed to write summary.json: {_decode.get('error', '')}")
        # DOC-END id=rm_index/RMIndex/build_and_save_summary/write_via_exec#1

        with self._lock:
            self.dir_summary = entries

        return summary_data
    # DOC-END id=rm_index/RMIndex/build_and_save_summary#1

    # DOC-BEGIN id=rm_index/RMIndex/load_summary#1 type=core v=1
    # summary: 从远程 .rm/summary.json 读取已有的目录索引到内存
    # intent: bind_workspace 时如果 .rm/summary.json 已存在，调用此方法直接加载，
    #   避免重新扫描。读取使用 history_manager.remote_read_file。
    def load_summary(self) -> dict:
        # DOC-BEGIN id=rm_index/RMIndex/load_summary/read_via_exec#1 type=behavior v=1
        # summary: 通过 _exec + cat 读取远程 summary.json，不再依赖 remote_read_file
        # intent: 与 load_file_docs 保持一致——通过 bound_terminal 直接读取，无需 task_id。
        #   summary.json 通常较小（<100KB），单次 cat 即可，不需要分段。
        cat_result = self._exec(f"cat '{self.rm_dir}/summary.json'")
        if not cat_result["ok"]:
            raise RuntimeError(f"Failed to read summary.json: {cat_result.get('error', '')}")
        result = {"ok": True, "content": cat_result["stdout"]}
        # DOC-END id=rm_index/RMIndex/load_summary/read_via_exec#1

        data = json.loads(result["content"])
        with self._lock:
            self.dir_summary = data.get("entries", {})
        return data
    # DOC-END id=rm_index/RMIndex/load_summary#1

    # DOC-BEGIN id=rm_index/RMIndex/load_file_docs#1 type=core v=1
    # summary: 按需加载指定文件的 DOC 块——从远程读取文件内容，解析 DOC 块，缓存到内存
    # intent: 懒加载策略的核心实现。只在 LLM 调用 open_summary 展开某个文件时才触发。
    #   解析后的 blocks 存入 self.file_docs[filepath]，同时写入 .rm/docs/{safe_name}.json
    #   作为持久化缓存。下次如果 .rm/docs/ 中已有且文件未修改，可以直接读取缓存（TODO: mtime 检查）。
    def load_file_docs(self, filepath: str) -> Dict[str, dict]:
        with self._lock:
            if filepath in self.file_docs:
                return self.file_docs[filepath]

        if filepath.startswith("/"):
            abs_path = filepath
        else:
            abs_path = f"{self.work_dir}/{filepath}"

        # DOC-BEGIN id=rm_index/RMIndex/load_file_docs/read_via_exec#1 type=behavior v=1
        # summary: 通过 _exec 直接读取远程文件内容（使用 cat 命令），不再依赖 remote_read_file
        # intent: remote_read_file 需要 task_id 来查找 bound terminal，但 RMIndex 在 workspace
        #   创建阶段可能没有有效的 task_id。改为直接通过 _exec（走 bound_terminal_id）执行 cat，
        #   与 RMIndex 的其他远程操作保持一致的执行路径。对于大文件使用 sed 分段读取。
        wc_result = self._exec(f"wc -l < '{abs_path}'")
        if not wc_result["ok"]:
            logger.warning(f"Failed to read {filepath}: {wc_result.get('error', '')}")
            return {}
        try:
            total_lines = int(wc_result["stdout"].strip())
        except ValueError:
            logger.warning(f"Cannot parse line count for {filepath}: {wc_result['stdout']}")
            return {}
        if total_lines == 0:
            file_content = ""
        else:
            chunk_lines = 800
            parts = []
            start = 1
            while start <= total_lines:
                end = start + chunk_lines - 1
                chunk_result = self._exec(f"sed -n '{start},{end}p' '{abs_path}'")
                if not chunk_result["ok"]:
                    logger.warning(f"Failed to read chunk {start}-{end} of {filepath}")
                    return {}
                parts.append(chunk_result["stdout"])
                start = end + 1
            file_content = "\n".join(parts)
        result = {"ok": True, "content": file_content}
        # DOC-END id=rm_index/RMIndex/load_file_docs/read_via_exec#1
        if not result["ok"]:
            logger.warning(f"Failed to read {filepath}: {result.get('error', '')}")
            return {}

        blocks = scan_file_docs(result["content"])

        with self._lock:
            self.file_docs[filepath] = blocks

        safe_name = filepath.replace("/", "__").replace(".", "_") + ".json"
        doc_data = {
            "file": filepath,
            "scan_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "blocks": {},
        }
        for bid, block in blocks.items():
            doc_data["blocks"][bid] = {
                "id": block["id"],
                "line_range": block["line_range"],
                "summary": block["summary"],
                "intent": block["intent"],
                "parent": block["parent"],
                "children": block["children"],
            }

        doc_content = json.dumps(doc_data, ensure_ascii=False, indent=2)
        # DOC-BEGIN id=rm_index/RMIndex/load_file_docs/write_cache#2 type=behavior v=2
        # summary: 通过 _exec + base64 将 DOC 块索引写入 .rm/docs/ 持久化缓存
        # intent: 不再依赖 remote_write_file（需要 task_id），改为直接通过 _exec 用 base64 安全写入。
        #   写缓存是锦上添花，失败时仅 warning 不阻断。
        import base64 as _b64
        _encoded = _b64.b64encode(doc_content.encode("utf-8")).decode("ascii")
        _cache_path = f"{self.rm_dir}/docs/{safe_name}"
        _tmp = f"/tmp/_rmdoc_{uuid.uuid4().hex[:8]}.b64"
        _wr = self._exec(f"echo -n '{_encoded}' > '{_tmp}' && base64 -d '{_tmp}' > '{_cache_path}' && rm -f '{_tmp}'")
        if not _wr.get("ok"):
            logger.warning(f"Failed to cache DOC index for {filepath}: {_wr.get('error', 'unknown')}")
        # DOC-END id=rm_index/RMIndex/load_file_docs/write_cache#2

        return blocks
    # DOC-END id=rm_index/RMIndex/load_file_docs#1

    # DOC-BEGIN id=rm_index/RMIndex/get_all_loaded_blocks#1 type=query v=1
    # summary: 返回当前内存中所有已加载的 DOC 块，按 {block_id: block_dict} 扁平化
    # intent: 供 SummaryStateManager 使用，需要快速查找任意 block_id 的信息。
    #   因为不同文件可能有相同 block_id（虽然规范要求唯一，但防御性编程），
    #   这里用 filepath::block_id 作为全局唯一键。
    def get_all_loaded_blocks(self) -> Dict[str, dict]:
        result = {}
        with self._lock:
            for filepath, blocks in self.file_docs.items():
                for bid, block in blocks.items():
                    global_key = f"{filepath}::{bid}"
                    result[global_key] = {**block, "_file": filepath}
        return result
    # DOC-END id=rm_index/RMIndex/get_all_loaded_blocks#1

    # DOC-BEGIN id=rm_index/RMIndex/read_file_content#1 type=query v=1
    # summary: 从远程读取文件原始内容并返回字符串，失败返回 None
    # intent: render_file_with_folding 需要文件原始内容来做代码折叠渲染。
    #   与 load_file_docs 中的读取逻辑相同，但此方法只返回内容不做 DOC 解析。
    #   使用 sed 分段读取以支持大文件。结果不做缓存——调用频率低（仅 open_summary 时触发），
    #   且文件可能随时被修改，缓存会导致内容过期。
    def read_file_content(self, filepath: str) -> Optional[str]:
        if filepath.startswith("/"):
            abs_path = filepath
        else:
            abs_path = f"{self.work_dir}/{filepath}"

        wc_result = self._exec(f"wc -l < '{abs_path}'")
        if not wc_result["ok"]:
            logger.warning(f"Failed to read {filepath}: {wc_result.get('error', '')}")
            return None
        try:
            total_lines = int(wc_result["stdout"].strip())
        except ValueError:
            logger.warning(f"Cannot parse line count for {filepath}: {wc_result['stdout']}")
            return None
        if total_lines == 0:
            return ""

        chunk_lines = 800
        parts = []
        start = 1
        while start <= total_lines:
            end = start + chunk_lines - 1
            chunk_result = self._exec(f"sed -n '{start},{end}p' '{abs_path}'")
            if not chunk_result["ok"]:
                logger.warning(f"Failed to read chunk {start}-{end} of {filepath}")
                return None
            parts.append(chunk_result["stdout"])
            start = end + 1
        return "\n".join(parts)
    # DOC-END id=rm_index/RMIndex/read_file_content#1

    # DOC-BEGIN id=rm_index/RMIndex/invalidate_file#1 type=behavior v=1
    # summary: 当文件被修改后，清除该文件的内存缓存，下次访问时强制重新扫描
    # intent: remote_write_file 写入文件后应调用此方法，确保 DOC 块索引与文件内容一致。
    #   只清除内存缓存，不删除 .rm/docs/ 中的持久化文件（下次 load 会覆盖）。
    def invalidate_file(self, filepath: str):
        with self._lock:
            self.file_docs.pop(filepath, None)
    # DOC-END id=rm_index/RMIndex/invalidate_file#1
# DOC-END id=rm_index/RMIndex#1