# src/services/agents.py

import glob
from typing import Callable, Dict, List, Tuple, Any, Optional, TypedDict, Literal
import re
import os
from pathlib import Path
import json
import base64
import uuid


class ToolCallsDict(TypedDict):
    tool_calls: list[Dict[str, Any]]
    num_of_trunc: int
    summarize_tool_call: list[Dict[str, Any]]

    @classmethod
    def create(cls) -> "ToolCallsDict":
        return {
            "tool_calls": [],
            "num_of_trunc": 0,
            "summarize_tool_call": [],
        }


class Tool_Calls:
    def __init__(
        self,
        LOG_DIR: str,
        MAX_CHAR: int,
        mode: str = Literal["Simple", "Summary"],
        UpdataFunc: Optional[Callable] = None,
    ):
        self.LOG_DIR = Path(LOG_DIR)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.PATH = Path(os.path.join(self.LOG_DIR, "tool_calls.json"))
        self.MAX_CHAR = MAX_CHAR
        self.mode = mode
        self.UpdataFunc = UpdataFunc

    def get_all_value(self) -> ToolCallsDict:
        if not self.PATH.exists():
            return ToolCallsDict.create()
        with open(self.PATH, "r", encoding="utf-8") as f:
            tool_calls_dict: ToolCallsDict = json.load(f)
            
        cleaned = []
        for msg in tool_calls_dict.get("tool_calls", []):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and "tool_calls" in msg
                and isinstance(msg["tool_calls"], list)
                and len(msg["tool_calls"]) == 0
            ):
                # 🔽 删除 tool_calls 字段，保留 content
                new_msg = dict(msg)
                new_msg.pop("tool_calls", None)
                cleaned.append(new_msg)
            else:
                cleaned.append(msg)
        tool_calls_dict["tool_calls"] = cleaned
        return tool_calls_dict

    def get_value(self) -> List[Dict[str, Any]]:
        if self.mode == "Simple":
            tool_calls_dict = self.get_all_value()
            return tool_calls_dict["tool_calls"]
        if self.mode == "Summary":
            return self.get_summerize_value()

    def get_summerize_value(self) -> List[Dict[str, Any]]:
        tool_calls_dict = self.get_all_value()
        tool_calls = tool_calls_dict["tool_calls"][tool_calls_dict["num_of_trunc"] :]
        if tool_calls_dict["num_of_trunc"] != 0:
            tool_calls = tool_calls_dict["summarize_tool_call"] + tool_calls
        return tool_calls

    def get_trunc_value(self) -> List[Dict[str, Any]]:
        all_msgs = self.get_value()
        selected: List[Dict[str, Any]] = []
        total_len = 0

        # 倒序遍历
        ll = 0
        for msg in reversed(all_msgs):
            msg_len = len(json.dumps(msg, ensure_ascii=False))
            total_len += msg_len
            selected.insert(0, msg)

            if msg.get("role") != "assistant":
                continue
            ll += 1

            if total_len > self.MAX_CHAR:
                print(f"\nTruncate the tool calls output {ll}!\n")
                break

        return selected

    def save(self, tool_calls_dict: ToolCallsDict):
        self.PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.PATH, "w", encoding="utf-8") as f:
            json.dump(tool_calls_dict, f, ensure_ascii=False, indent=2)

    def extend(self, new_tool_calls: List[Dict[str, Any]]):
        tool_calls_dict = self.get_all_value()
        tool_calls_dict["tool_calls"].extend(new_tool_calls)
        self.save(tool_calls_dict)

    def insert_summarize_tool_calls(
        self, tool_calls: List[Dict[str, Any]], num_of_trunc: int
    ):
        tool_calls_dict = self.get_all_value()
        tool_calls_dict["num_of_trunc"] = num_of_trunc
        tool_calls_dict["summarize_tool_call"] = tool_calls
        self.save(tool_calls_dict)

    def clear(self):
        self.save(ToolCallsDict.create())

    @classmethod
    def summarize_tool_calls(cls, tool_calls: List[Dict[str, Any]]) -> str:
        """
        Summarize a list of tool call records into a concise string.

        Args:
            tool_calls: list of dicts, each representing a tool call.
        """
        lines = []

        def truncate(text: str) -> str:
            text = text.strip().replace("\n", " ")
            return text

        for item in tool_calls:
            role = item.get("role", "")
            name = item.get("name", "")
            index = item.get("index", "?")

            # Case 1: assistant generating a tool call
            if "tool_calls" in item:
                for call in item["tool_calls"]:
                    func = call.get("function", {})
                    func_name = func.get("name", "[unknown]")
                    args = func.get("arguments", {})
                    args_str = truncate(json.dumps(args, ensure_ascii=False))
                    idx = call.get("index", "?")
                    lines.append(f"[{idx}] call {func_name}({args_str})")

            # Case 2: tool’s response
            elif role == "tool":
                content = truncate(item.get("content", ""))
                lines.append(f"[→] {name}: {content}")

        return "\n".join(lines)
    
    def collapse_new_tool_calls_into_summary(self) -> List[Dict[str, Any]]:
        """
        Collapse tool calls from [num_of_trunc : end] into simplified assistant messages,
        append them into summarize_tool_call, and advance num_of_trunc.
        """
        tool_calls_dict = self.get_all_value()
        all_msgs = tool_calls_dict["tool_calls"]

        start = tool_calls_dict["num_of_trunc"]
        new_msgs = all_msgs[start:]

        if not new_msgs:
            return []

        # 1️⃣ tool_call_id -> tool result
        tool_results: Dict[str, Dict[str, Any]] = {}
        for msg in new_msgs:
            if msg.get("role") == "tool":
                tool_results[msg.get("tool_call_id")] = {
                    "name": msg.get("name", "[unknown]"),
                    "content": msg.get("content", "")
                }

        collapsed_messages: List[Dict[str, Any]] = []

        # 2️⃣ collapse assistant tool_calls
        for msg in new_msgs:
            if msg.get("role") != "assistant":
                continue
            if "tool_calls" not in msg:
                continue

            llm_content = (msg.get("content") or "").strip()

            for call in msg["tool_calls"]:
                call_id = call.get("id")
                func = call.get("function", {})

                tool_name = func.get("name", "[unknown]")
                arguments = func.get("arguments", {})

                result = tool_results.get(call_id, {})
                result_content = result.get("content", "[no result]")

                parts = []

                # ✅ 原始 LLM response
                if llm_content:
                    parts.append(
                        "llm response:\n"
                        f"{llm_content}"
                    )

                # ✅ tool 调用
                parts.append(
                    f"Use {tool_name}\n"
                    f"with variable:\n"
                    f"{json.dumps(arguments, ensure_ascii=False, indent=2)}"
                )

                # ✅ tool 结果
                parts.append(
                    "get results:\n"
                    f"{result_content}"
                )

                collapsed_messages.append({
                    "role": "assistant",
                    "content": "\n\n".join(parts),
                })

        # 3️⃣ 写入 summarize_tool_call（这是关键修正点）
        tool_calls_dict["summarize_tool_call"].extend(collapsed_messages)

        # 4️⃣ 推进 trunc cursor
        tool_calls_dict["num_of_trunc"] = len(all_msgs)

        # 5️⃣ 保存
        self.save(tool_calls_dict)

        return collapsed_messages
    
    def dehydrate_content(self, target_content: str, hash_val: str, filename: str):
        """
        【历史脱水】
        遍历所有历史记录，如果发现 content 中包含 target_content，
        则将其替换为脱水标记。
        """
        tool_calls_dict = self.get_all_value()
        all_msgs = tool_calls_dict["tool_calls"]
        summarize_msgs = tool_calls_dict.get("summarize_tool_call", [])
        
        changed = False

        # 1. 处理主历史
        for msg in all_msgs:
            if msg.get("content") and target_content in msg["content"]:
                # 使用特殊的脱水格式，告知 LLM 这里的代码已经变成悬挂状态
                placeholder = f"\n[CODE_STAGED: {filename} | HASH: {hash_val} | CONTENT_OMITTED]\n"
                msg["content"] = msg["content"].replace(target_content, placeholder)
                changed = True
        
        # 2. 处理已折叠的摘要 (Summary 模式)
        for msg in summarize_msgs:
            if msg.get("content") and target_content in msg["content"]:
                placeholder = f"\n[CODE_STAGED: {filename} | HASH: {hash_val}]\n"
                msg["content"] = msg["content"].replace(target_content, placeholder)
                changed = True

        if changed:
            self.save(tool_calls_dict)
        return changed

    def inject_pinned_content(self, hash_val: str, filename: str, content: str):
        """
        【历史注入】
        在历史记录的最后手动插入一条记录，告知 LLM 该 Hash 关联的当前完整代码。
        这种情况通常发生在：
        1. 代码刚被挂起时
        2. LLM 询问该 Hash 具体内容时
        """
        # 我们使用 assistant 角色发送一个“环境同步”消息
        injection_msg = {
            "role": "assistant",
            "content": (
                f"[SYSTEM_SYNC]\n"
                f"The following content is now Pinned and associated with Hash: {hash_val}\n"
                f"File: {filename}\n"
                f"--- CONTENT START ---\n"
                f"{content}\n"
                f"--- CONTENT END ---\n"
                f"Use the § Start/End/Replace protocol to propose edits to this hash."
            )
        }
        self.extend([injection_msg])

    def force_refresh_hash(self, old_hash: str, new_hash: str, new_content: str, filename: str):
        """
        【状态更新】
        当代码修改成功并产生新 Hash 时：
        1. 将记录中的所有旧 Hash 占位符更新为新 Hash (可选)
        2. 注入当前的新鲜内容
        """
        # 这通常结合 inject_pinned_content 使用
        self.inject_pinned_content(new_hash, filename, new_content)
        
    # DOC-BEGIN id=agents/tool_calls/trim_edit_protocol_blocks#1 type=behavior v=1
    # summary: 方法trim_edit_protocol_blocks遍历 tool_calls 与 summarize_tool_call 两类历史消息，
    #   对每条消息的content执行正则扫描；若发现完整的“§ Start/§ End/§ Replace”三段式补丁协议块，
    #   则将该协议块整体替换为占位符字符串[EDIT_PROTOCOL_TRIMMED]；返回值为bool，表示是否发生过实际替换
    # intent: trim_history=True 场景下，历史里已经“应用成功”的补丁全文对后续推理价值很低，却极耗 token；
    #   用占位符保留“发生过编辑”的语义线索即可。实现上用DOTALL跨行匹配，并用非贪婪匹配避免吞掉多段补丁；
    #   同时仅在content包含关键标记(§ Start/§ Replace)时才进入正则，降低扫描成本与误伤概率。
    def trim_edit_protocol_blocks(self) -> bool:
        import re

        data = self.get_all_value()
        changed = False

        # DOC-BEGIN id=agents/tool_calls/trim_edit_protocol_blocks/pair_regex#1 type=behavior v=2
        # summary: 正则匹配"行首 § 开头的行"构成的配对块：从第一个行首 § 行开始，到下一个行首 § 行（含该行末尾换行）结束，
        #   整体（含两行 § 行自身及其换行符）作为一个匹配单元被替换为占位符
        # intent: 实际 LLM 输出的补丁格式是 § Start...§ Start / § End...§ End / § Replace...§ Replace，
        #   共 6 个 § 行构成 3 对。用逐对匹配更鲁棒：即使只输出了部分补丁也能清理。
        #   使用 re.MULTILINE 让 ^ 匹配每行开头，确保 § 必须在行首，避免误匹配行内的 § 字符。
        #   ^§[^\n]*\n 精确吃掉整行（含末尾换行），中间 .*? 非贪婪跨行匹配到最近的下一个行首 § 行。
        #   最后的 (?:\n|$) 确保第二个 § 行的换行也被吃掉，避免替换后留下多余空行，
        #   同时 $ 兜底处理 § 行位于文本末尾无换行的情况。
        pair_pattern = re.compile(
            r"^§[^\n]*\n"       # 第一个行首 § 行（整行 + 换行符）
            r".*?"              # 中间内容（非贪婪，跨行）
            r"^§[^\n]*"         # 第二个行首 § 行（整行）
            r"(?:\n|$)",        # 第二个 § 行的换行符（或文本结尾）
            re.DOTALL | re.MULTILINE,
        )

        for msg_list_key in ("tool_calls", "summarize_tool_call"):
            for msg in data.get(msg_list_key, []):
                content = msg.get("content")
                if not content or not isinstance(content, str):
                    continue
                if "§" not in content:
                    continue

                new_content, n = pair_pattern.subn("[EDIT_PROTOCOL_TRIMMED]", content)
                if n > 0 and new_content != content:
                    msg["content"] = new_content
                    changed = True

        if changed:
            self.save(data)
        return changed
    # DOC-END id=agents/tool_calls/trim_edit_protocol_blocks#1

    # DOC-BEGIN id=agents/tool_calls/trim_redundant_code_blocks#1 type=behavior v=1
    # summary: 扫描历史消息中所有 markdown 代码块（```...```），如果某个代码块的内容
    #   被任一 pinned_code 的 content 完全包含（子串关系），则将该代码块替换为占位符，
    #   返回是否有实际修改
    # intent: pinned_code 已作为 TEMPORARY_CONTEXT 注入到最新消息中，历史中出现的
    #   与其内容重复的代码块浪费 token。判断条件是"pinned content 包含该代码块内容"
    #   而非严格相等，因为历史中的代码块可能是文件的一部分片段。
    #   使用 strip() 后的子串匹配避免空白差异导致的误判。
    #   长度阈值 50 字符避免替换过短的代码片段（如单行示例）导致语义丢失。
    def trim_redundant_code_blocks(self, pinned_codes: list):
        """
        将历史中被 pinned_codes 完全覆盖的代码块替换为占位符。
        
        pinned_codes: list of objects with .hash, .filename, .content attributes
        """
        import re
        if not pinned_codes:
            return False

        # 预处理 pinned content（strip 后缓存）
        pinned_contents = []
        for p in pinned_codes:
            pinned_contents.append({
                "content_stripped": p.content.strip(),
                "hash": p.hash,
                "filename": p.filename,
            })

        code_block_pattern = re.compile(r'```[^\n]*\n(.*?)```', re.DOTALL)

        data = self.get_all_value()
        changed = False

        for msg_list_key in ("tool_calls", "summarize_tool_call"):
            for msg in data.get(msg_list_key, []):
                content = msg.get("content")
                if not content or not isinstance(content, str):
                    continue
                if "```" not in content:
                    continue

                def replacer(match):
                    nonlocal changed
                    block_content = match.group(1).strip()
                    # 跳过过短的代码块
                    if len(block_content) < 50:
                        return match.group(0)
                    for pc in pinned_contents:
                        if block_content in pc["content_stripped"]:
                            changed = True
                            return f"[CODE_BLOCK_TRIMMED: content covered by pinned file {pc['filename']} (hash: {pc['hash']})]"
                    return match.group(0)

                new_content = code_block_pattern.sub(replacer, content)
                if new_content != content:
                    msg["content"] = new_content

        if changed:
            self.save(data)
        return changed
    # DOC-END id=agents/tool_calls/trim_redundant_code_blocks#1

    def remove_temporary_context(self):
        """
        物理删除所有带有 [TEMPORARY_CONTEXT] 标记的消息
        """
        data = self.get_all_value()
        original_len = len(data["tool_calls"])
        
        # 过滤掉所有包含标记的消息
        data["tool_calls"] = [
            msg for msg in data["tool_calls"] 
            if not (isinstance(msg.get("content"), str) and "[TEMPORARY_CONTEXT]" in msg["content"])
        ]
        
        if len(data["tool_calls"]) != original_len:
            self.save(data)
            return True
        return False

    def inject_temporary_context(self, pinned_codes: List[Any]):
        """
        将当前的挂载代码作为一条新的临时消息存入磁盘
        """
        if not pinned_codes:
            return

        blocks = []
        for p in pinned_codes:
            blocks.append(f"File: {p.filename} (Hash: {p.hash})\n```\n{p.content}\n```")
        
        # DOC-BEGIN id=agents/tool_calls/inject_temporary_context/insert_before_last_user#1 type=behavior v=2
        # summary: 临时上下文消息使用 role="user" 构造，并插入到 tc 历史中最后一条真实 user 消息之前
        # intent: Anthropic Claude 要求对话必须以 user 消息结尾，不支持 assistant prefill。
        #   如果把临时上下文 append 到 tc 末尾，虽然 role 是 user 但它不是用户的真实问题，
        #   语义上不合理且可能干扰 LLM 理解对话结构。正确做法是保持用户真实消息始终在最后，
        #   将辅助上下文插入到它之前。使用 _insert_before_last_user_msg 实现此逻辑。
        tmp_msg = {
            "role": "user",
            "content": (
                "[TEMPORARY_CONTEXT] Current Pinned Files:\n" + 
                "\n\n".join(blocks) + 
                "\n[End of Temporary Context]"
            )
        }
        # DOC-END id=agents/tool_calls/inject_temporary_context/insert_before_last_user#1
        self._insert_before_last_user_msg(tmp_msg)

    # DOC-BEGIN id=agents/tool_calls/_insert_before_last_user_msg#1 type=core v=1
    # summary: 将一条消息插入到 tc 历史中最后一条 role="user" 消息之前；若没有 user 消息则 append 到末尾
    # intent: workspace summary 和 temporary context 都需要在"用户真实消息之前"注入，
    #   这样 tc 末尾始终是用户的真实提问，兼容 Anthropic Claude 等要求对话以 user 消息结尾的模型。
    #   从后往前找第一条 user 消息的位置，在该位置之前 insert。如果历史中没有 user 消息
    #   （理论上不应发生，因为对话总是由 user 发起），则 fallback 为 append。
    def _insert_before_last_user_msg(self, msg: dict):
        data = self.get_all_value()
        msgs = data["tool_calls"]
        
        insert_idx = len(msgs)  # fallback: append
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user" and "[TEMPORARY_CONTEXT]" not in msgs[i].get("content", "") and "[WORKSPACE_SUMMARY]" not in msgs[i].get("content", ""):
                insert_idx = i
                break
        
        msgs.insert(insert_idx, msg)
        self.save(data)
    # DOC-END id=agents/tool_calls/_insert_before_last_user_msg#1

    # DOC-BEGIN id=agents/tool_calls/remove_workspace_summary#1 type=behavior v=1
    # summary: 物理删除 tc 历史中所有带 [WORKSPACE_SUMMARY] 标记的消息，返回是否有实际删除
    # intent: workspace summary 是临时注入的——每个 LLM step 前注入最新版本、step 后删除。
    #   它不应持久存储在 tc 中，因为每次 step 后项目状态都可能变化（文件编辑、summary 节点展开/关闭、
    #   function calling 执行远程命令等）。使用独立标记 [WORKSPACE_SUMMARY] 区分于
    #   [TEMPORARY_CONTEXT]（pinned code 用），两者生命周期不同：pinned code 跨 step 持久，
    #   workspace summary 每 step 重建。
    def remove_workspace_summary(self) -> bool:
        data = self.get_all_value()
        original_len = len(data["tool_calls"])
        data["tool_calls"] = [
            msg for msg in data["tool_calls"]
            if not (isinstance(msg.get("content"), str) and "[WORKSPACE_SUMMARY]" in msg["content"])
        ]
        if len(data["tool_calls"]) != original_len:
            self.save(data)
            return True
        return False
    # DOC-END id=agents/tool_calls/remove_workspace_summary#1

    # DOC-BEGIN id=agents/tool_calls/inject_workspace_summary#3 type=behavior v=3
    # summary: 将 workspace summary 文本作为一条 user 消息插入到 tc 历史中最后一条真实 user 消息之前
    # intent: 与 inject_temporary_context 保持一致的注入策略——辅助上下文始终在真实 user 消息之前，
    #   确保 tc 末尾是用户的真实提问，兼容 Anthropic Claude 等不支持 assistant prefill 的模型。
    #   消息内容以 [WORKSPACE_SUMMARY] 开头，供 remove_workspace_summary 识别和删除。
    def inject_workspace_summary(self, summary_text: str):
        if not summary_text:
            return
        msg = {
            "role": "user",
            "content": summary_text,
        }
        self._insert_before_last_user_msg(msg)
    # DOC-END id=agents/tool_calls/inject_workspace_summary#3


def generate_call_id():
    random_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("utf-8").rstrip("=")
    return f"call_00_{random_id[:22]}"


def get_func_tool_call(
    func_name: str, result: Optional[str] = None, **args
) -> List[Dict[str, Any]]:
    """
    Generate a tool call entry that mimics the structure of assistant/tool messages.

    Args:
        func_name (str): The name of the function/tool being called.
        result (str): The result/output returned by the function/tool.
        **args: Arbitrary keyword arguments representing the function arguments.

    Returns:
        List[Dict[str, Any]]: A list containing two messages:
            1. assistant message with the function call in `tool_calls`
            2. tool message with the actual function result
    """
    call_id = generate_call_id()
    if not isinstance(args, str):
        args_str = args
    else:
        args_str = json.dumps(args)

    # Assistant message
    assistant_msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": func_name, "arguments": args_str},
                "index": 0,
                "name": func_name,
            }
        ],
    }

    # Tool message
    if result is None:
        assistant_msg = assistant_msg["tool_calls"]
    else:
        tool_msg = {
            "role": "tool",
            "name": func_name,
            "tool_call_id": call_id,
            "content": result,
        }
        assistant_msg = [assistant_msg, tool_msg]

    return assistant_msg


def summarize_tools(tools: List[Callable]) -> str:
    """
    Generate a text summary of tools and their descriptions.
    """
    lines = []
    for tool in tools:
        name = tool.__name__
        desc = tool.__doc__.strip() if tool.__doc__ else "No description available."
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def stop_if_no_tool_calls(*, tool_calls) -> bool:
    """
    Stop condition for agent loop.

    Stop when the model did NOT request any tool calls.

    Args:
        tool_calls: list | None
            Parsed tool_calls from the current LLM step.

    Returns:
        bool: True if should stop, False otherwise.
    """
    # None 或 空 list 都视为「没有工具调用」
    if not tool_calls:
        return True
    return False