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
        
        tmp_msg = {
            "role": "assistant",
            "content": (
                "[TEMPORARY_CONTEXT] Current Pinned Files:\n" + 
                "\n\n".join(blocks) + 
                "\n[End of Temporary Context]"
            )
        }
        # 直接持久化到磁盘
        self.extend([tmp_msg])


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