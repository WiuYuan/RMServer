# src/services/llm.py

import requests
import json
import inspect
import logging
from typing import Callable, List, Dict, Any, Union, Optional
from src.utils.prompts import get_prompt
from src.services.agents import Tool_Calls, get_func_tool_call
import copy
import base64

# from src.services.agents import Tool
from src.utils.summary_attention_dag import SummaryAttentionDAG
import os
import uuid
from pathlib import Path
from collections import defaultdict
from src.services.external_client import ExternalClient
import re


os.environ["NO_PROXY"] = "*"

logger = logging.getLogger(__name__)


class LLM:
    def __init__(
        self,
        api_key: str = "",
        llm_url: str = "http://localhost:11434/api/chat",
        model_name: str = "qwen3:8b",
        proxies: dict = None,
        format: str = "ollama",
        system_prompt: str = "",
        ec: Optional[ExternalClient] = None,
        reasoning_enabled: bool = False,
    ):
        """
        Initialize the LLM instance.

        Parameters:
        llm_url (str): The URL of the LLM service (e.g., Ollama API endpoint).
        model_name (str): The model to use. Default is "qwen3:32b".
        reasoning_enabled (bool): Whether to enable model reasoning output (for OpenRouter/other models supporting separate reasoning field). Default is False.
        """
        self.api_key = api_key
        self.llm_url = llm_url
        self.model_name = model_name
        self.proxies = proxies or {"http": None, "https": None}
        self.format = format
        self.system_prompt = system_prompt
        self.ec = ec
        self.reasoning_enabled = reasoning_enabled


    def query(self, prompt: str, verbose: bool = True) -> str:
        # if verbose:
        #     # print("[LLM] Query")
        #     self.ec.send_message(
        #         {
        #             "type": "info",
        #             "data": {
        #                 "category": "LLM",
        #                 "content": "Query",
        #                 "level_delta": 1,
        #             },
        #         }
        #     )
        #     formatted_prompt = prompt.replace("\n", "\n    ")
        #     formatted_system_prompt = self.system_prompt.replace("\n", "\n    ")
        #     # print(f"  [SYSTEM PROMPT]\n    {formatted_system_prompt}")
        #     self.ec.send_message(
        #         {
        #             "type": "info",
        #             "data": {
        #                 "category": "SYSTEM PROMPT",
        #                 "content": formatted_system_prompt,
        #                 "level_delta": 0,
        #             },
        #         }
        #     )
        #     # print(f"  [INPUT]\n    {formatted_prompt}")
        #     self.ec.send_message(
        #         {
        #             "type": "info",
        #             "data": {
        #                 "category": "INPUT",
        #                 "content": formatted_prompt,
        #                 "level_delta": 0,
        #             },
        #         }
        #     )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            result = self.query_messages(messages, verbose=verbose)
            if self.ec is not None:
                self.ec.send_message(None)
            return result
        finally:
            # 🔚 唯一的 EOS：通知 consumer 本次流式输出结束
            if self.ec is not None:
                self.ec.send_message(None)
        # if verbose:
        #     self.ec.send_message(
        #         {
        #             "type": "info",
        #             "data": {
        #                 "category": "",
        #                 "content": "",
        #                 "level_delta": -1,
        #             },
        #         }
        #     )

    # DOC-BEGIN id=llm/query_with_tools#2 type=api v=3
    # summary: 多轮工具调用主循环，接收 system_prompt/prompt/tools 等参数，循环调用 LLM 并执行工具，
    #   支持 output_interceptor（每 step 文本输出后解析 § 协议）、pre_step_hook（每 step 前注入临时上下文）、
    #   post_step_hook（对话结束后清理临时上下文）
    # intent: pre_step_hook/post_step_hook 是 workspace bound 模式的核心机制——
    #   summary 作为临时消息在每个 step 前注入 tc、step 后删除再重新注入，
    #   确保 LLM 每一步都能看到最新的项目结构和文件内容。
    #   三个 hook 都是 Optional[Callable]，不传或传 None 时行为与原来完全一致。
    #   output_interceptor 解析 § Start/End/Replace 和 § Touch/Write 并同步远程文件。
    #   pre_step_hook 在 LLM 调用前执行（注入最新 summary），post_step_hook 在循环退出后执行（清理 summary）。
    def query_with_tools(
        self,
        system_prompt: Union[str, Callable[[], str]],
        prompt: str,
        max_steps: int,
        tc: Tool_Calls,
        extra_guide_tool_call: List[Dict[str, Any]] = [],
        tools: List[Callable] = None,
        verbose: bool = True,
        stop_condition: Callable[..., bool] = None,
        check_start_prompt: str = None,
        tool_runner: Callable = None,
        output_interceptor: Callable = None,
        pre_step_hook: Callable = None,
        post_step_hook: Callable = None,
    ) -> str:
    # DOC-END id=llm/query_with_tools#2
        # DOC-BEGIN id=llm/query_with_tools/func_dict#1 type=behavior v=1
        # summary: 从 tools 列表构建 {函数名: 函数对象} 字典，供后续 tool call 分发使用
        # intent: 必须在循环开始前构建，否则 tool call 处理时 func_dict 未定义导致 NameError。
        #   与 query_with_tools_by_attention 保持一致的构建方式。
        func_dict = {func.__name__: func for func in tools} if tools else {}
        # DOC-END id=llm/query_with_tools/func_dict#1

        # DOC-BEGIN id=llm/query_with_tools/should_stop_helper#1 type=behavior v=1
        # summary: 辅助函数——通过 tool_runner(None, None) 查询当前 stop 信号状态
        # intent: tool_runner 传入 func=None 时立即返回 (None, should_stop())，不执行任何工具。
        #   封装为 _check_stop() 供主循环 step 开头和其他非 tool-call 位置复用，
        #   避免到处写 tool_runner is not None + tool_runner(None, None) 的重复判断。
        #   如果 tool_runner 为 None（无取消机制），始终返回 False。
        def _check_stop() -> bool:
            if tool_runner is not None:
                _, stopped = tool_runner(None, None)
                return stopped
            return False
        # DOC-END id=llm/query_with_tools/should_stop_helper#1

        messages = [
            {"role": "user", "content": prompt}
        ]
        system_messages = [
            {"role": "system", "content": system_prompt() if callable(system_prompt) else system_prompt}
        ]
        tc.extend(messages)
        all_texts = []
        for step in range(max_steps):
            # resample = False
            print(f"\n=== Prompt step {step+1} ===\n")

            # DOC-BEGIN id=llm/query_with_tools/early_stop_check#1 type=behavior v=1
            # summary: 每个 step 开头检查 stop 信号，若已被请求停止则立即退出主循环
            # intent: 上一个 step 的 tool call 可能在执行完成后才收到 stop 信号（时序竞争），
            #   此时 should_stop 在 tool_calls 处理阶段未触发，控制流回到 for 循环顶部。
            #   如果不在 step 开头检查，会白白发起一次新的 LLM API 请求（数秒延迟 + token 消耗），
            #   直到流式读取循环中的 tool_runner(None, None) 才能检测到 stop。
            #   此检查确保 stop 信号在最早时机生效。
            if _check_stop():
                if verbose:
                    print(f"\n=== Stopping at step {step+1} by user (early check) ===\n")
                tc.extend([{
                    "role": "assistant",
                    "content": "\n".join(all_texts) + "\n\nStop By User!" if all_texts else "Stop By User!",
                }])
                break
            # DOC-END id=llm/query_with_tools/early_stop_check#1

            # DOC-BEGIN id=llm/query_with_tools/pre_step_hook#1 type=behavior v=1
            # summary: 每个 step 开始时调用 pre_step_hook，用于注入最新的 workspace summary 到 tc
            # intent: pre_step_hook 由 llm_handlers 构造，内部先删除旧 summary 再注入新 summary。
            #   放在 system_prompt 刷新之前，因为 system_prompt 可能引用 tc 中的内容（如终端快照），
            #   而 summary 注入到 tc 后 system_prompt 闭包可以感知到最新状态。
            #   异常被捕获并 warning，不中断主循环。
            if pre_step_hook is not None:
                try:
                    pre_step_hook()
                except Exception as e:
                    import traceback as _tb
                    logger.warning(f"pre_step_hook error: {e}\n{_tb.format_exc()}")
            # DOC-END id=llm/query_with_tools/pre_step_hook#1

            if callable(system_prompt):
                system_messages[0]["content"] = system_prompt()
            text = ""
            if check_start_prompt is None:
                text, tool_calls, should_stop = self.query_messages_with_tools(
                    system_messages + tc.get_value() + extra_guide_tool_call,
                    tools=tools,
                    tool_runner=tool_runner,
                    verbose=verbose,
                )
            else:
                while not text.startswith(check_start_prompt):
                    text, tool_calls, should_stop = self.query_messages_with_tools(
                        system_messages + tc.get_value() + extra_guide_tool_call,
                        tools=tools,
                        tool_runner=tool_runner,
                        verbose=verbose,
                    )
                    if not text.startswith(check_start_prompt):
                        print(f"\nCheck Failed, Generate Again!\n")

            all_texts.append(text)

            # DOC-BEGIN id=llm/query_with_tools/call_interceptor#1 type=behavior v=1
            # summary: 每个 step 的 LLM 文本输出完成后，调用 output_interceptor 回调解析 § 编辑协议
            # intent: 放在 tool_calls 处理之前、should_stop 检查之前，确保即使用户中途 stop，
            #   已输出的编辑指令也能被解析执行。interceptor 内部异常被捕获并 warning，不中断主循环。
            if output_interceptor is not None and text.strip():
                try:
                    output_interceptor(text)
                except Exception as e:
                    import traceback as _tb
                    logger.warning(f"output_interceptor error: {e}\n{_tb.format_exc()}")
            # DOC-END id=llm/query_with_tools/call_interceptor#1

            if should_stop:
                if verbose:
                    print(f"\n=== Stopping at step {step+1} by user ===\n")
                tc.extend([
                    {
                        "role": "assistant",
                        "content": text + "\n\nStop By User!",
                    }
                ])
                return "\n".join(all_texts)
                
            new_tool_calls = [
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls,
                }
            ]

            for call in tool_calls:
                if self.format == "ollama":
                    call = call["function"]
                func_name = call["name"]
                args = call["function"]["arguments"]

                if func_name in func_dict:
                    if isinstance(args, str):
                        args = json.loads(args)
                    if verbose:
                        formatted_args = self._format_arguments_for_display(
                            func_name, args
                        )
                        formatted_args = formatted_args.replace("\n", "\n    ")
                        print(
                            f"\nCalling function '{func_name}' with arguments:\n{formatted_args}"
                        )
                        self.ec.send_message(
                                {
                                    "type": "llm_info",
                                    "data": {
                                        "content": f"Calling function '{func_name}' with arguments:\n{formatted_args}\n",
                                    },
                                }
                            )
                    if tool_runner is None:
                        result = func_dict[func_name](**args)
                        should_stop = False
                    else: 
                        result, should_stop = tool_runner(func_dict[func_name], args)
                    self.ec.send_message(
                        {
                            "type": "llm_info",
                            "data": {
                                "content": f"Complete calling function '{func_name}'.\n",
                            },
                        }
                    )
                else:
                    result = f"Function {func_name} not found"
                if should_stop:
                    if verbose:
                        print(f"\n=== Stopping at step {step+1} by user ===\n")
                    tc.extend([
                        {
                            "role": "assistant",
                            "content": text + "\n\nStop By User!",
                        }
                    ])
                    return "\n".join(all_texts)
                new_tool_calls.append(
                    {
                        "role": "tool",
                        "name": func_name,
                        "tool_call_id": call.get("id", ""),
                        "content": str(result),
                    }
                )
            tc.extend(new_tool_calls)
            if tc.UpdataFunc is not None:
                tc.UpdataFunc()
            if stop_condition and stop_condition(tool_calls=tool_calls):
                if verbose:
                    print(f"\n=== Stopping at step {step+1} ===\n")
                break

        # DOC-BEGIN id=llm/query_with_tools/post_step_hook#1 type=behavior v=1
        # summary: 主循环结束后调用 post_step_hook，清理 tc 中残留的 workspace summary 临时消息
        # intent: 无论是正常结束（stop_condition）还是达到 max_steps，都需要清理 summary，
        #   避免 summary 残留在 tc 的持久化历史中（summary 是临时的，不应持久存储）。
        if post_step_hook is not None:
            try:
                post_step_hook()
            except Exception as e:
                import traceback as _tb
                logger.warning(f"post_step_hook error: {e}\n{_tb.format_exc()}")
        # DOC-END id=llm/query_with_tools/post_step_hook#1

        return "\n".join(all_texts)

    def query_messages(self, messages: str, verbose: bool = True) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            # "logprobs": True,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        payload = {k: v for k, v in payload.items() if v is not None}
        # print(messages)

        text_accumulate = ""

        # if self.model_name == "human":
        #     open(
        #         "/Users/yuanwen/Desktop/Docker_Environment/intern2/2/test_prompt.txt",
        #         "w",
        #         encoding="utf-8",
        #     ).write(prompt)
        #     option = input()
        #     if option == "1":
        #         raise RuntimeError("exit")
        #     text_accumulate = open(
        #         "/Users/yuanwen/Desktop/Docker_Environment/intern2/2/test_answer.txt",
        #         "r",
        #     ).read()
        #     return text_accumulate

        # Make a streaming POST request
        # if verbose:
        #     # print(f"  [OUTPUT]\n    ", end="", flush=True)
        #     self.ec.send_message(
        #         {
        #             "type": "info",
        #             "data": {
        #                 "category": "OUTPUT",
        #                 "content": "",
        #                 "level_delta": 1,
        #             },
        #         }
        #     )
        with requests.post(
            self.llm_url,
            headers=headers,
            json=payload,
            proxies=self.proxies,
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                # print(line)
                line_str = line.decode("utf-8").strip()
                if self.format == "ollama":
                    chunk = json.loads(line_str)
                    token = None
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            print(token.replace("\n", "\n    "), end="", flush=True)
                            # self.ec.send_message(
                            #     {
                            #         "type": "llm_info",
                            #         "data": {
                            #             "content": token,
                            #         },
                            #     }
                            # )

                if self.format == "openai":
                    line_str = line_str[len("data: ") :]
                    if line_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON from line: {line_str}")
                        # if line_str.startswith("-alive"):
                        continue
                        # print(messages)
                        # raise
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        token = None
                        reasoning_token = None
                        if "content" in delta:
                            token = delta["content"]
                        if "reasoning" in delta:
                            reasoning_token = delta["reasoning"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            print(token.replace("\n", "\n    "), end="", flush=True)
                            # self.ec.send_message(
                            #     {
                            #         "type": "llm_info",
                            #         "data": {
                            #             "content": token,
                            #         },
                            #     }
                            # )
                    # 处理推理内容
        return text_accumulate

    def _format_arguments_for_display(self, func_name: str, args: dict) -> str:
        """
        Format function arguments for better display, especially for code content.

        Parameters:
            func_name (str): The name of the function being called
            args (dict): The arguments dictionary

        Returns:
            str: Formatted string representation of arguments
        """
        formatted_args = []

        for key, value in args.items():
            if isinstance(value, str):
                # Special formatting for text/code content in write functions
                if key in ["text", "new_text", "old_text", "info"]:
                    # If it looks like code (contains newlines), format it nicely
                    if "\n" in value:
                        formatted_value = f"\n{'-'*40}\n{value}\n{'-'*40}"
                    else:
                        formatted_value = repr(value)
                else:
                    formatted_value = repr(value)
            else:
                formatted_value = repr(value)

            formatted_args.append(f"  {key}: {formatted_value}")

        return "\n".join(formatted_args)

    def query_messages_with_tools(
        self,
        messages: str,
        tools: Union[str, Callable[[], str]] = None,
        verbose: bool = True,
        tool_runner: Callable = None,
    ) -> tuple[str, list, bool]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        tools = tools or []
        tools = self.create_tools(tools)

        for msg in messages:
            if "tool_calls" in msg:
                for call in msg["tool_calls"]:
                    if "function" in call and "arguments" in call["function"]:
                        args = call["function"]["arguments"]
                        if not isinstance(args, str):
                            # 只有在 dict/非 str 的时候才转为 JSON 字符串
                            call["function"]["arguments"] = json.dumps(args)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream": True,
            # "logprobs": True,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        payload = {k: v for k, v in payload.items() if v is not None}

        text_accumulate = ""

        # Make a streaming POST request
        tool_calls = []
        # if verbose:
        #     self.ec.send_message(
        #         {
        #             "type": "info",
        #             "data": {
        #                 "category": "OUTPUT",
        #                 "content": "",
        #                 "level_delta": 1,
        #             },
        #         }
        #     )
        #     # print(f"  [OUTPUT]\n    ", end="", flush=True)
        with requests.post(
            self.llm_url,
            headers=headers,
            json=payload,
            proxies=self.proxies,
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if tool_runner is not None:
                    _, should_stop = tool_runner(None, None)
                    if should_stop:
                        response.close()
                        return text_accumulate, tool_calls, should_stop
                if not line:
                    continue
                # print(line)
                line_str = line.decode("utf-8").strip()
                if self.format == "ollama":
                    chunk = json.loads(line_str)
                    token = None
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]

                    if token:
                        text_accumulate += token
                        if verbose:
                            self.ec.send_message(
                                {
                                    "type": "llm_info",
                                    "data": {
                                        "content": token,
                                    },
                                }
                            )
                            print(token.replace("\n", "\n    "), end="", flush=True)

                    if "message" in chunk and "tool_calls" in chunk["message"]:
                        tool_calls.extend(chunk["message"]["tool_calls"])

                if self.format == "openai":
                    line_str = line_str[len("data: ") :]
                    if line_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line_str)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON from line: {line_str}")
                        if verbose:
                            self.ec.send_message(
                                {
                                    "type": "llm_info",
                                    "data": {
                                        "content": "",
                                    },
                                }
                            )
                        # if line_str.startswith("-alive"):
                        continue
                        # print(messages)
                        # raise
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        token = None
                        reasoning_token = None
                        if "content" in delta:
                            token = delta["content"]
                        if "reasoning" in delta:
                            reasoning_token = delta["reasoning"]
                        if "tool_calls" in delta:
                            tool_calls.extend(delta["tool_calls"])
                            # print(chunk)

                    if token:
                        text_accumulate += token
                        if verbose:
                            self.ec.send_message(
                                {
                                    "type": "llm_info",
                                    "data": {
                                        "content": token,
                                    },
                                }
                            )
                            print(token.replace("\n", "\n    "), end="", flush=True)
                    # 处理推理内容，单独发送给前端区分展示
                    if reasoning_token and verbose and self.ec is not None:
                        self.ec.send_message(
                            {
                                "type": "llm_reasoning",
                                "data": {
                                    "content": reasoning_token,
                                },
                            }
                        )
                        # 控制台打印推理内容用灰色标识方便区分
                        print(reasoning_token.replace('\n', '\n    '), end="", flush=True)

        if self.format == "openai":
            grouped = defaultdict(list)
            for call in tool_calls:
                idx = call.get("index", 0)
                args = call.get("function", {}).get("arguments", "")
                grouped[idx].append(args)

            tool_calls_clean = []
            for idx, parts in grouped.items():
                full_args_str = "".join(parts).strip()
                try:
                    full_args = json.loads(full_args_str) if full_args_str else {}
                except json.JSONDecodeError:
                    full_args = full_args_str

                func_name = None
                fields = ["id", "type", "function"]
                extracted = {}
                for call in tool_calls:
                    if call.get("index") == idx:
                        if "name" in call.get("function", {}):
                            func_name = call["function"]["name"]
                        for field in fields:
                            if field in call:
                                extracted[field] = call.get(field)
                        break

                new_tool_call = {
                    **extracted,
                    "index": idx,
                    "name": func_name,
                }
                new_tool_call["function"]["arguments"] = full_args

                tool_calls_clean.append(new_tool_call)
            tool_calls = tool_calls_clean

        # if verbose:
        #     self.ec.send_message(
        #         {
        #             "type": "info",
        #             "data": {
        #                 "category": "",
        #                 "content": "",
        #                 "level_delta": -1,
        #             },
        #         }
        #     )
        print("")

        return text_accumulate, tool_calls, False

    @classmethod
    def create_tools(cls, func_list: List[Callable]) -> List[Dict[str, Any]]:
        tools = []

        for func in func_list:
            sig = inspect.signature(func)
            func_name = func.__name__
            func_description = func.__doc__ or f"Function {func_name}"
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in (int, float):
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    else:
                        param_type = "string"

                param_description = f"Parameter {param_name}"

                properties[param_name] = {
                    "type": param_type,
                    "description": param_description,
                }

                if param.default == inspect.Parameter.empty:
                    required.append(param_name)

            tool = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_description.strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
            tools.append(tool)

        return tools

    # DOC-BEGIN id=llm/image-processing#1 type=utility v=1
    # summary: 图片处理工具方法，用于优化LLM处理图片的成本
    # intent: 高分辨率图片直接发送给LLM会消耗大量token，需要预处理降低分辨率

    @staticmethod
    def reduce_image_resolution(base64_image: str, scale_factor: float = 0.5) -> str:
        """
        降低图片分辨率，减少LLM token消耗。

        Parameters:
        base64_image (str): base64编码的图片字符串（可带data URL前缀）
        scale_factor (float): 缩放比例（默认0.5，即缩小到50%）

        Returns:
        str: 缩放后的base64图片字符串（带data URL前缀）
        """
        from PIL import Image
        import io as _io

        # 处理data URL前缀
        if base64_image.startswith('data:'):
            header, data = base64_image.split(',', 1)
            img_bytes = base64.b64decode(data)
        else:
            header = 'data:image/jpeg;base64'
            img_bytes = base64.b64decode(base64_image)

        img = Image.open(_io.BytesIO(img_bytes))
        original_width, original_height = img.size
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = _io.BytesIO()
        if 'png' in header:
            fmt = 'PNG'
        elif 'webp' in header:
            fmt = 'WEBP'
        else:
            fmt = 'JPEG'

        img_resized.save(buffer, format=fmt)
        resized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return f"{header},{resized_base64}"

    def query_multimodal(self, prompt: str, images: List[str], verbose: bool = True) -> str:
        """
        多模态查询，支持文本和图片输入，无需历史记录，非流式。
        每张图片前插入 "--- FIG x ---" 标签，让模型明确知道图片编号。

        Parameters:
        prompt (str): 文本提示
        images (List[str]): 图片base64字符串列表（格式：data:image/jpeg;base64,...）
        verbose (bool): 是否打印输出

        Returns:
        str: LLM响应文本
        """
        content = []
        content.append({"type": "text", "text": prompt})

        for idx, img_base64 in enumerate(images, start=1):
            if not img_base64.startswith('data:'):
                img_base64 = f"data:image/jpeg;base64,{img_base64}"
            content.append({
                "type": "text",
                "text": f"--- FIG {idx} ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": img_base64}
            })
            content.append({
                "type": "text",
                "text": f"--- FIG {idx} ---"
            })

        messages = [{"role": "user", "content": content}]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}

        try:
            response = requests.post(
                self.llm_url,
                headers=headers,
                json=payload,
                proxies=self.proxies,
            )
            response.raise_for_status()
            
            # DOC-BEGIN id=llm/query_multimodal/json_parse#1 type=robustness v=1
            # summary: 安全解析响应 JSON，处理可能的不完整或格式错误响应
            # intent: 部分 LLM 服务可能返回不完整的 JSON（如截断的响应），直接调用 response.json() 可能抛出 JSONDecodeError。
            #   使用 response.text 手动解析，并在解析失败时返回详细的错误信息，包括响应状态码和内容片段。
            try:
                result = response.json()
            except json.JSONDecodeError as json_err:
                error_msg = f"JSON解析失败: {json_err}, 响应状态码: {response.status_code}, 响应内容前500字符: {response.text[:500]}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            # DOC-END id=llm/query_multimodal/json_parse#1

            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0].get('message', {}).get('content', '')
                if verbose:
                    print(f"[LLM multimodal] response length={len(text)}")
                return text

            return ""
        except Exception as e:
            logger.error(f"Multimodal query failed: {e}")
            return f"Error: {e}"

    # DOC-END id=llm/image-processing#1
