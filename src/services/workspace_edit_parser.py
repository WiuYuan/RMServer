# src/services/workspace_edit_parser.py
import re
import logging
from typing import Dict, Callable, Optional

logger = logging.getLogger(__name__)


# DOC-BEGIN id=workspace_edit_parser/module#1 type=design v=1
# summary: 解析 LLM 输出文本中的 § 编辑协议（Start/End/Replace 和 Touch/Write），
#   将解析结果同步到远程文件系统
# intent: workspace bound 模式下，LLM 的代码编辑不再依赖前端 pinned code 机制，
#   而是由后端拦截 LLM 的每一步文本输出，解析其中的 § 协议指令并直接写入远程文件。
#   支持两类协议：
#     1. § Start/End/Replace (hash) —— 对已有文件的局部编辑（锚点匹配替换）
#     2. § Touch (path) + § Write (path) —— 新建文件或完整覆写文件
#   hash_to_path 映射由调用方（llm_handlers）构建并传入。
# DOC-END id=workspace_edit_parser/module#1


# DOC-BEGIN id=workspace_edit_parser/parse_start_end_replace#1 type=core v=1
# summary: 从 LLM 输出文本中解析所有 § Start/End/Replace 编辑指令，返回结构化编辑列表
# intent: 正则匹配三段式协议：§ Start (hash) ... § Start (hash) 定义起始锚点，
#   § End (hash) ... § End (hash) 定义结束锚点，§ Replace (hash) ... § Replace (hash) 定义替换内容。
#   三段的 hash 必须一致才视为有效编辑。锚点内容会剥离 markdown 代码块标记（```）。
#   一次 LLM 输出中可能包含多个独立的编辑指令（针对不同文件或同一文件的不同位置）。
def parse_start_end_replace(text: str) -> list:
    edits = []
    pattern = re.compile(
        r'§\s*Start\s*\((\w+)\)\s*\n'       # § Start (hash)
        r'(.*?)\n'                             # anchor content (may contain ```)
        r'§\s*Start\s*\(\1\)\s*\n'            # § Start (hash) closing
        r'\s*'
        r'§\s*End\s*\(\1\)\s*\n'              # § End (hash)
        r'(.*?)\n'                             # anchor content
        r'§\s*End\s*\(\1\)\s*\n'              # § End (hash) closing
        r'\s*'
        r'§\s*Replace\s*\(\1\)\s*\n'          # § Replace (hash)
        r'(.*?)\n'                             # replacement content
        r'§\s*Replace\s*\(\1\)',               # § Replace (hash) closing
        re.DOTALL
    )
    for m in pattern.finditer(text):
        hash_val = m.group(1)
        start_anchor = _strip_code_fence(m.group(2))
        end_anchor = _strip_code_fence(m.group(3))
        replacement = _strip_code_fence(m.group(4))
        edits.append({
            "type": "start_end_replace",
            "hash": hash_val,
            "start_anchor": start_anchor,
            "end_anchor": end_anchor,
            "replacement": replacement,
        })
    return edits
# DOC-END id=workspace_edit_parser/parse_start_end_replace#1


# DOC-BEGIN id=workspace_edit_parser/parse_touch_write#1 type=core v=1
# summary: 从 LLM 输出文本中解析所有 § Touch 和 § Write 指令，返回结构化文件操作列表
# intent: § Touch (path) 表示创建空文件或确保文件存在（类似 unix touch），
#   § Write (path) ... § Write (path) 表示完整覆写文件内容。
#   path 是相对于工作区根目录的路径（或绝对路径），不是 hash。
#   这两个指令用于新建文件场景，因为新文件没有 hash 映射。
# DOC-BEGIN id=workspace_edit_parser/parse_touch_write#2 type=core v=2
# summary: 从 LLM 输出文本中解析所有 § Touch 和 § Write 指令，返回结构化文件操作列表；
#   § Write 同时支持代码块包裹（```lang ... ```）和无代码块两种格式
# intent: § Touch (path) 表示创建空文件或确保文件存在（类似 unix touch），
#   § Write (path) ... § Write (path) 表示完整覆写文件内容。
#   LLM 倾向于用 markdown 代码块包裹代码内容（更易读），但有时也会省略代码块。
#   解析策略：先用通用模式匹配 § Write 开闭标签之间的所有内容，
#   然后用 _strip_code_fence 剥离可能存在的代码块标记。
#   这样一套正则同时兼容两种格式，无需分别匹配。
def parse_touch_write(text: str) -> list:
    ops = []
    
    # § Touch (path)
    touch_pattern = re.compile(r'§\s*Touch\s*\(([^)]+)\)')
    for m in touch_pattern.finditer(text):
        path = m.group(1).strip()
        ops.append({"type": "touch", "path": path})
    
    # § Write (path) ... § Write (path)
    # DOC-BEGIN id=workspace_edit_parser/parse_touch_write/write_pattern#1 type=behavior v=1
    # summary: 匹配 § Write 开闭标签之间的全部内容，然后由 _strip_code_fence 处理可选的代码块包裹
    # intent: 正则只负责定位 § Write 边界，不区分有无代码块——这由后处理统一处理。
    #   支持的格式举例：
    #     格式1（带代码块）: § Write (path)\n```python\ncontent\n```\n§ Write (path)
    #     格式2（无代码块）: § Write (path)\ncontent\n§ Write (path)
    #   两种格式经过 _strip_code_fence 后都得到纯 content。
    write_pattern = re.compile(
        r'§\s*Write\s*\(([^)]+)\)\s*\n'
        r'(.*?)\n'
        r'§\s*Write\s*\(\1\)',
        re.DOTALL
    )
    # DOC-END id=workspace_edit_parser/parse_touch_write/write_pattern#1
    for m in write_pattern.finditer(text):
        path = m.group(1).strip()
        content = _strip_code_fence(m.group(2))
        ops.append({"type": "write", "path": path, "content": content})
    
    return ops
# DOC-END id=workspace_edit_parser/parse_touch_write#2


# DOC-BEGIN id=workspace_edit_parser/_strip_code_fence#1 type=util v=1
# summary: 剥离 markdown 代码块标记（开头的 ```xxx 和结尾的 ```），保留纯代码内容
# intent: LLM 输出的 § 协议中，锚点和替换内容通常包裹在 markdown 代码块中，
#   但实际匹配和写入时需要纯代码。只剥离首尾各一个 ``` 行，中间的 ``` 保留。
def _strip_code_fence(text: str) -> str:
    lines = text.split('\n')
    if lines and re.match(r'^```\w*$', lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines)
# DOC-END id=workspace_edit_parser/_strip_code_fence#1


# DOC-BEGIN id=workspace_edit_parser/apply_start_end_replace#1 type=core v=1
# summary: 对单个文件内容执行一条 Start/End/Replace 编辑——找到锚点位置并替换中间内容
# intent: 锚点匹配使用 str.find（精确子串匹配，不是正则），与前端 pinned code 的行为一致。
#   start_anchor 必须在文件中恰好出现一次，end_anchor 也必须恰好出现一次且在 start 之后。
#   如果匹配失败（0次或多次），返回 (None, error_msg) 而不是静默跳过，调用方应记录日志。
#   替换范围包含两个锚点本身（即 start_anchor 的第一个字符到 end_anchor 的最后一个字符）。
def apply_start_end_replace(content: str, edit: dict) -> tuple:
    start_anchor = edit["start_anchor"].strip()
    end_anchor = edit["end_anchor"].strip()
    replacement = edit["replacement"]

    # 查找 start_anchor
    start_count = content.count(start_anchor)
    if start_count != 1:
        return None, f"Start anchor found {start_count} times (expected 1)"
    
    # 查找 end_anchor
    end_count = content.count(end_anchor)
    if end_count != 1:
        return None, f"End anchor found {end_count} times (expected 1)"
    
    start_idx = content.find(start_anchor)
    end_idx = content.find(end_anchor)
    
    if end_idx < start_idx:
        return None, "End anchor appears before start anchor"
    
    # 替换范围：从 start_anchor 开头到 end_anchor 结尾
    end_of_end = end_idx + len(end_anchor)
    new_content = content[:start_idx] + replacement + content[end_of_end:]
    
    return new_content, None
# DOC-END id=workspace_edit_parser/apply_start_end_replace#1


# DOC-BEGIN id=workspace_edit_parser/make_interceptor#1 type=api v=1
# summary: 工厂函数——构造 output_interceptor 闭包，供 LLM query_with_tools 主循环每步调用
# intent: 闭包捕获 hash_to_path 映射、file_reader 和 file_writer 三个依赖，
#   返回的 interceptor(text) 函数依次解析 § Start/End/Replace 和 § Touch/Write，
#   对每条指令执行相应的远程文件操作。所有异常在内部捕获并 warning，不抛出。
#   file_reader: (path) -> Optional[str]，读取远程文件内容
#   file_writer: (path, content) -> bool，写入远程文件内容
# DOC-BEGIN id=workspace_edit_parser/parse_exec#1 type=core v=1
# summary: 从 LLM 输出文本中解析 § Exec ... § Exec 块，提取其中的单条命令（verb + args）
# intent: § Exec 块内部用 ``` 包裹一条命令，每次只允许一条。
#   命令格式为 "verb arg1 arg2 ..."，当前支持的 verb 只有 push。
#   push 的语法为 "push <remote_path> as <display_name>"。
#   返回解析后的结构化指令列表（通常只有一个元素）。
#   如果 § Exec 块内有多行命令，只取第一条非空行（协议约定每次一条）。
def parse_exec(text: str) -> list:
    cmds = []
    pattern = re.compile(
        r'§\s*Exec\s*\n'
        r'(.*?)\n'
        r'§\s*Exec',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        body = _strip_code_fence(m.group(1)).strip()
        if not body:
            continue
        # 取第一条非空行
        line = None
        for l in body.split('\n'):
            l = l.strip()
            if l:
                line = l
                break
        if not line:
            continue
        # 解析 push <path> as <name>
        push_match = re.match(r'^push\s+(.+?)\s+as\s+(\S+)$', line)
        if push_match:
            cmds.append({
                "verb": "push",
                "remote_path": push_match.group(1).strip(),
                "name": push_match.group(2).strip(),
            })
        else:
            logger.warning(f"[parse_exec] Unknown command: {line}")
    return cmds
# DOC-END id=workspace_edit_parser/parse_exec#1


# DOC-BEGIN id=workspace_edit_parser/ext_name_to_mime#1 type=util v=1
# summary: 根据文件扩展名推断 MIME 类型，默认 application/octet-stream
# intent: push 指令需要将 MIME 类型传递给前端，前端据此决定渲染方式（<img> vs 下载链接等）。
#   只覆盖常见图片格式，其余走默认值。
_EXT_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".pdf": "application/pdf",
}

def _ext_to_mime(name: str) -> str:
    import os
    ext = os.path.splitext(name)[1].lower()
    return _EXT_TO_MIME.get(ext, "application/octet-stream")
# DOC-END id=workspace_edit_parser/ext_name_to_mime#1


# DOC-BEGIN id=workspace_edit_parser/make_interceptor#2 type=api v=2
# summary: 工厂函数——构造 output_interceptor 闭包，支持 § Start/End/Replace、§ Touch/Write、§ Exec push
# intent: 新增 ec（ExternalClient）和 binary_reader 两个可选参数。
#   当 § Exec push 被检测到时，通过 binary_reader 读取远程二进制文件，
#   base64 编码后通过 ec.send_image 推送到前端。
#   ec 和 binary_reader 仅在 workspace bound 模式下传入，非 workspace 场景不受影响。
#   binary_reader 签名: (path: str) -> dict，返回 {"ok": True, "data": bytes, ...} 或 {"ok": False, ...}
def make_interceptor(
    hash_to_path: Dict[str, str],
    file_reader: Callable[[str], Optional[str]],
    file_writer: Callable[[str, str], bool],
    ec=None,
    binary_reader: Callable[[str], dict] = None,
) -> Callable[[str], None]:
    
    def interceptor(text: str) -> None:
        # 1. 处理 § Start/End/Replace
        edits = parse_start_end_replace(text)
        for edit in edits:
            h = edit["hash"]
            path = hash_to_path.get(h)
            if not path:
                logger.warning(f"[interceptor] Unknown hash {h}, skipping edit")
                continue
            try:
                current = file_reader(path)
                if current is None:
                    logger.warning(f"[interceptor] Cannot read {path} for hash {h}")
                    continue
                new_content, err = apply_start_end_replace(current, edit)
                if err:
                    logger.warning(f"[interceptor] Edit failed for {path}: {err}")
                    continue
                ok = file_writer(path, new_content)
                if ok:
                    logger.info(f"[interceptor] Applied edit to {path} (hash {h})")
                else:
                    logger.warning(f"[interceptor] Failed to write {path}")
            except Exception as e:
                logger.warning(f"[interceptor] Exception editing {path}: {e}")

        # 2. 处理 § Touch / § Write
        file_ops = parse_touch_write(text)
        for op in file_ops:
            try:
                if op["type"] == "touch":
                    ok = file_writer(op["path"], "")
                    if ok:
                        logger.info(f"[interceptor] Touched {op['path']}")
                elif op["type"] == "write":
                    ok = file_writer(op["path"], op["content"])
                    if ok:
                        logger.info(f"[interceptor] Wrote {op['path']}")
            except Exception as e:
                logger.warning(f"[interceptor] Exception in file op {op['type']} {op['path']}: {e}")

        # DOC-BEGIN id=workspace_edit_parser/interceptor_exec_push#1 type=behavior v=1
        # summary: 解析 § Exec 块中的 push 命令，读取远程二进制文件并通过 ec.send_image 推送到前端
        # intent: 仅当 ec 和 binary_reader 都存在时才处理（非 workspace 场景跳过）。
        #   push 指令的 remote_path 可以是绝对路径或相对路径（binary_reader 内部处理转换）。
        #   读取失败时 warning 并跳过，不影响 LLM 输出流。
        #   base64 编码在此处完成（binary_reader 返回 raw bytes），而非让 binary_reader 返回 base64，
        #   保持 binary_reader 的通用性。
        if ec and binary_reader:
            exec_cmds = parse_exec(text)
            for cmd in exec_cmds:
                if cmd["verb"] == "push":
                    try:
                        result = binary_reader(cmd["remote_path"])
                        if result.get("ok") and result.get("data"):
                            import base64 as b64_mod
                            b64_str = b64_mod.b64encode(result["data"]).decode("ascii")
                            mime = _ext_to_mime(cmd["name"])
                            ec.send_image(
                                name=cmd["name"],
                                data_base64=b64_str,
                                mime_type=mime,
                            )
                            logger.info(f"[interceptor] Pushed image {cmd['name']} ({len(result['data'])} bytes)")
                        else:
                            logger.warning(f"[interceptor] Failed to read binary file {cmd['remote_path']}: {result.get('error', 'unknown')}")
                    except Exception as e:
                        logger.warning(f"[interceptor] Exception in push {cmd['remote_path']}: {e}")
        # DOC-END id=workspace_edit_parser/interceptor_exec_push#1
    
    return interceptor
# DOC-END id=workspace_edit_parser/make_interceptor#2
# DOC-END id=workspace_edit_parser/make_interceptor#1