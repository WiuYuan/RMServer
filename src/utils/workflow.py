from src.services.llm import LLM, load_messages, save_messages
from src.services.logical_tree import LogicalTree
from typing import List, Tuple, Callable, Dict, Any, Union
from src.services.custom_tools import custom_tools
from src.utils.prompts import get_prompt
from src.services.agents import Tool_Calls, get_func_tool_call, summarize_tools
from src.services.external_client import ExternalClient
import json
import uuid
import base64
import os


def generate_prompt_for_solve_goal_list(
    goal: str, tree: LogicalTree, tree_filepath: str
) -> str:
    """
    Generate an English prompt for the LLM to solve a task step by step
    using a logical tree structure.

    Args:
        goal (str): The overall goal to achieve.
        tree (LogicalTree): The current logical tree, which includes the current node.

    Returns:
        str: A formatted prompt for the LLM.
    """
    tree_json = tree.to_json()
    current_node = tree.current_node

    prompt = get_prompt("solve_task_with_logical_tree_prompt.txt").format(
        goal=goal,
        tree_json=tree_json,
        current_node_name=current_node.name,
        current_node_description=current_node.description.replace("\n", "\n    "),
    )
    tree.save(tree_filepath)
    return prompt.strip()


def solve_goal_list(
    goal_list: List[Tuple[str, str]],
    ct: custom_tools,
    max_steps: int,
    tc: Tool_Calls,
    verbose: bool,
    human_decision: bool = False,
    tools: Union[List[Callable], None] = None,
    human_check_before_calling: bool = False,
):
    tree_filepath = os.path.join(tc.LOG_DIR, "logical_tree.json")

    tree = LogicalTree("Pseudo Root", "Pseudo Root Description")

    def stop_condition(*args, **kwargs):
        return tree.current_node == tree.root

    def help_add_child(info: str) -> str:
        """
        Add a new child node based on the given information.

        This function helps create a new subtask (child node) in the logical tree.
        The input `info` should contain all information gathered so far that is relevant
        to the logic of the task. The function will use this information to call an LLM
        and determine the appropriate child node to add.

        Args:
            info (str): Relevant information accumulated so far to guide the creation
                        of a new child task.
        """
        if verbose:
            print(f"\n[FUNC] Start handling add child.\n")
        tree_json = tree.to_json()
        current_node = tree.current_node
        prompt = get_prompt("help_add_child_prompt.txt").format(
            tree_json=tree_json,
            current_node_name=current_node.name,
            current_node_description=current_node.description,
            info=info.replace("\n", "\t\n"),
        )
        tools_add_child = [tree.add_child, ct.func_guide]
        func_dict = {func.__name__: func for func in tools_add_child}
        messages = [{"role": "user", "content": prompt}]
        check_start_prompt = "I see the help add child guidance"
        guidance = get_prompt("extra_help_add_child_prompt.txt").format(
            check_start_prompt=check_start_prompt
        )
        extra_guide_tool_call = get_func_tool_call(
            func_name="func_guide", result=guidance, guidance="Will Return Guidance"
        )
        while True:
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "LLM",
                            "content": "Query with Tools",
                            "level_delta": 1,
                        },
                    }
                )
            # print(f"[LLM] Query with Tools Start")
            # formatted_prompt = prompt.replace("\n", "\n    ")
            # print(f"  [INPUT]\n    {formatted_prompt}")
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "INPUT",
                            "content": prompt,
                            "level_delta": 0,
                        },
                    }
                )
            text, tool_calls = ct.llm.query_messages_with_tools(
                messages + extra_guide_tool_call,
                tools=tools_add_child,
                verbose=verbose,
            )
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "",
                            "content": "",
                            "level_delta": -1,
                        },
                    }
                )
            if text.startswith(check_start_prompt) == False:
                print(f"\nCheck Text Start Failed, Generate Again!\n")
                continue
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "SAMPLE",
                            "content": "Sampling Next Steps",
                            "level_delta": 1,
                        },
                    }
                )
            # print("\n[SAMPLE] Sampling Next Steps")
            sample_next_step = ["Resample"]
            for call in tool_calls:
                if ct.llm.format == "ollama":
                    call = call["function"]
                func_name = call["name"]
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                if verbose:
                    formatted_args = ct.llm._format_arguments_for_display(
                        func_name, args
                    )
                    print(
                        f"\n  Sample will call function '{func_name}' with arguments:\n{formatted_args}"
                    )
                    sample_next_step.append(
                        f"\n  Sample will call function '{func_name}' with arguments:\n{formatted_args}"
                    )
            if human_decision:
                while True:
                    try:
                        # num = int(
                        #     input(
                        #         "Please choose one as next step (0 represents regenerate again): "
                        #     )
                        # )
                        next_option = ct.llm.ec.get_choice_response(
                            question="Please choose one as next step (0 represents regenerate again): ",
                            options=sample_next_step,
                        )
                        num = 0
                        for i in range(len(sample_next_step)):
                            if sample_next_step[i] == next_option:
                                num = i

                        if 0 <= num <= len(tool_calls):
                            break
                        else:
                            print(
                                f"❌ Invalid input. Please enter a number between 0 and {len(tool_calls)}."
                            )
                    except ValueError:
                        print("❌ Invalid input. Please enter a valid integer.")

                if num == 0:
                    if verbose:
                        ct.llm.ec.send_message(
                            {
                                "type": "info",
                                "data": {
                                    "category": "",
                                    "content": "",
                                    "level_delta": -1,
                                },
                            }
                        )
                    continue
                tool_calls = [tool_calls[num - 1]]
            else:
                guidance = (
                    f"Below are the sample tool calls\n"
                    f"{json.dumps(tool_calls, ensure_ascii=False, indent=2)}\n"
                    "You must choose only **ONE** as your next step tool call."
                )
                choose_add_child_tool_call = get_func_tool_call(
                    func_name="func_guide",
                    result="Will Return Guidance",
                    guidance=guidance,
                )
                if verbose:
                    ct.llm.ec.send_message(
                        {
                            "type": "info",
                            "data": {
                                "category": "LLM",
                                "content": "Select Automatically",
                                "level_delta": 1,
                            },
                        }
                    )
                # print(f"[LLM] Query with Tools Start")
                # formatted_prompt = prompt.replace("\n", "\n    ")
                # print(f"  [INPUT]\n    {formatted_prompt}")
                if verbose:
                    ct.llm.ec.send_message(
                        {
                            "type": "info",
                            "data": {
                                "category": "INPUT",
                                "content": prompt,
                                "level_delta": 0,
                            },
                        }
                    )
                text, tool_calls = ct.llm.query_messages_with_tools(
                    messages + tc.get_value() + choose_add_child_tool_call,
                    tools=tools_add_child,
                    verbose=verbose,
                )
                if verbose:
                    ct.llm.ec.send_message(
                        {
                            "type": "info",
                            "data": {
                                "category": "",
                                "content": "",
                                "level_delta": -1,
                            },
                        }
                    )
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "",
                            "content": "",
                            "level_delta": -1,
                        },
                    }
                )
            new_tool_calls = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                }
            ]

            for call in tool_calls:
                if ct.llm.format == "ollama":
                    call = call["function"]
                func_name = call["name"]
                args = call["function"]["arguments"]

                if func_name in func_dict:
                    if isinstance(args, str):
                        args = json.loads(args)
                    if verbose:
                        formatted_args = ct.llm._format_arguments_for_display(
                            func_name, args
                        )
                        print(
                            f"\n[CALL] function '{func_name}' with arguments:\n{formatted_args}"
                        )
                    result = func_dict[func_name](**args)
                else:
                    result = f"Function {func_name} not found"

                new_tool_calls.append(
                    {
                        "role": "tool",
                        "name": func_name,
                        "tool_call_id": call.get("id", ""),
                        "content": str(result),
                    }
                )
            tc.extend(new_tool_calls)
            if verbose:
                print(f"\n[FUNC] Complete handling add child.\n")
            break
        return "Successfully run help_add_child!"

    def help_return_to_parent(info: str):
        """
        Decide whether to return to the parent node or add a new child node
        based on the given information.

        This function helps manage the logical tree by using the accumulated
        information in `info`. It will call an LLM to determine whether the
        current task (node) is already completed. If the task is complete,
        it calls `return_to_parent`. Otherwise, it calls `help_add_child` to
        add a new subtask (child node).

        Args:
            info (str): Relevant information accumulated so far to guide
                        the decision of whether to return to parent or add
                        a new child task.
        """
        if verbose:
            print(f"\n[FUNC] Start handling return to parent.\n")
        info = ct.func_extract_factual_info(info)

        tree_json = tree.to_json()
        current_node = tree.current_node
        tool_calls = []
        if human_decision:
            decision = input(
                "If you think the task is completed, please input 'Yes', otherwise input your guidance"
            )
            if decision == "Yes":
                prompt = f"Human think the below info support back:\n{info}"
                tools_return_to_parent = [tree.return_to_parent]
                tool_calls.extend(
                    get_func_tool_call(
                        func_name="return_to_parent", text=prompt, success=True
                    )
                )
            else:
                guidance = f"Human give the following guidance:{decision}\n"
                tools_return_to_parent = [ct.func_guide]
                tool_calls.extend(
                    get_func_tool_call(
                        func_name="func_guide",
                        result=None,
                        guidance=guidance,
                    )
                )
        else:
            # if current_node.back_num >= 0:
            #     prompt = get_prompt("help_return_to_parent_resample_prompt.txt").format(
            #         tree_json=tree_json,
            #         current_node_name=current_node.name,
            #         current_node_description=current_node.description,
            #         info=info.replace("\n", "\t\n"),
            #     )
            #     tools_return_to_parent = [help_add_child]
            # else:
            prompt = get_prompt("help_return_to_parent_prompt.txt").format(
                tree_json=tree_json,
                current_node_name=current_node.name,
                current_node_description=current_node.description,
                info=info.replace("\n", "\t\n"),
            )
            tools_return_to_parent = [tree.return_to_parent, ct.func_guide]
            # current_node.back_num -= 1
        func_dict = {func.__name__: func for func in tools_return_to_parent}
        if len(tool_calls) == 0:
            messages = [{"role": "user", "content": prompt}]
            print(f"[LLM] Query with Tools Start")
            formatted_prompt = prompt.replace("\n", "\n    ")
            print(f"  [INPUT]\n    {formatted_prompt}")
            text, tool_calls = ct.llm.query_messages_with_tools(
                messages,
                tools=tools_return_to_parent,
                verbose=verbose,
            )
        new_tool_calls = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            }
        ]
        for call in tool_calls:
            if ct.llm.format == "ollama":
                call = call["function"]
            func_name = call["name"]
            args = call["function"]["arguments"]

            if func_name in func_dict:
                if isinstance(args, str):
                    args = json.loads(args)
                if verbose:
                    formatted_args = ct.llm._format_arguments_for_display(
                        func_name, args
                    )
                    print(
                        f"\n[CALL] function '{func_name}' with arguments:\n{formatted_args}"
                    )
                result = func_dict[func_name](**args)
            else:
                result = f"Function {func_name} not found"

            new_tool_calls.append(
                {
                    "role": "tool",
                    "name": func_name,
                    "tool_call_id": call.get("id", ""),
                    "content": str(result),
                }
            )
        tc.extend(new_tool_calls)
        if verbose:
            print(f"\n[FUNC] Complete handling return to parent.\n")
        return "Successfully run help_return_to_parent!"

    def UpdataFunc():
        tool_calls = tc.get_value()
        if len(json.dumps(tool_calls)) > tc.MAX_CHAR:
            id = min(15, int(len(tool_calls) * 2 / 3))
            summary_tool_calls = query_summary_for_tool_calls(
                ct.llm, tc, id, tc.MAX_CHAR / 5
            )
            tc.insert_summarize_tool_calls(summary_tool_calls, id)

    # tc.UpdataFunc = UpdataFunc
    tc.mode = "Simple"
    if tools is None:
        tools = [
            ct.func_cat,
            ct.func_cat_with_llm,
            ct.func_write,
            ct.func_modify,
            ct.func_insert,
            ct.func_append,
            ct.func_prepend,
            ct.func_python,
            ct.func_cmd,
            ct.func_matlab,
            ct.func_fetch_info_from_pmc_with_llm,
            ct.func_human,
            ct.func_guide,
            # ct.func_think,
            ct.add_child,
            ct.return_to_parent,
            help_add_child,
            help_return_to_parent,
        ]
    else:
        tools.extend(
            [
                help_add_child,
                help_return_to_parent,
            ]
        )

    for idx, (goal_name, goal_description) in enumerate(goal_list, start=1):
        result = tree.add_child(goal_name, goal_description)
        tc.extend(
            get_func_tool_call(
                "add_child",
                result=result,
                goal_name=goal_name,
                goal_description=goal_description,
            )
        )

        if verbose:
            print(f"\n[INFO] Starting LLM query for goal {idx}...\n")

        guidance = get_prompt("tools_emphasize_prompt.txt").format(
            tool_descriptions=summarize_tools(tools).replace("\n", "\t\n"),
        )
        extra_guide_tool_call = get_func_tool_call(
            func_name="func_guide", result=guidance, guidance="Will Return Guidance"
        )
        extra_guide_tool_call = []
        guidance = get_prompt("extra_guidance_prompt.txt").format(
            current_node_name=tree.current_node.name,
            current_node_description=tree.current_node.description,
        )
        extra_guide_tool_call.extend(
            get_func_tool_call(
                func_name="func_guide", result=guidance, guidance="Will Return Guidance"
            )
        )

        text = ct.llm.query_with_tools_by_attention(
            prompt=lambda: generate_prompt_for_solve_goal_list(
                goal_description, tree, tree_filepath
            ),
            max_steps=max_steps,
            tools=tools,
            tc=tc,
            extra_guide_tool_call=extra_guide_tool_call,
            verbose=verbose,
            stop_condition=stop_condition,
            check_start_prompt="I see the guidance",
            human_check_before_calling=human_check_before_calling,
        )

        if verbose:
            print(f"\n[INFO] Finished processing goal {idx}\n")



def solve_goal_list_test(
    goal_list: List[Tuple[str, str]],
    ct: custom_tools,
    max_steps: int,
    tc: Tool_Calls,
    verbose: bool,
    human_decision: bool = False,
    tools: Union[List[Callable], None] = None,
    human_check_before_calling: bool = False,
):
    tree_filepath = os.path.join(tc.LOG_DIR, "logical_tree.json")

    tree = LogicalTree("Pseudo Root", "Pseudo Root Description")

    def stop_condition(*args, **kwargs):
        return tree.current_node == tree.root

    def help_add_child(info: str) -> str:
        """
        Add a new child node based on the given information.

        This function helps create a new subtask (child node) in the logical tree.
        The input `info` should contain all information gathered so far that is relevant
        to the logic of the task. The function will use this information to call an LLM
        and determine the appropriate child node to add.

        Args:
            info (str): Relevant information accumulated so far to guide the creation
                        of a new child task.
        """
        if verbose:
            print(f"\n[FUNC] Start handling add child.\n")
        tree_json = tree.to_json()
        current_node = tree.current_node
        prompt = get_prompt("help_add_child_prompt.txt").format(
            tree_json=tree_json,
            current_node_name=current_node.name,
            current_node_description=current_node.description,
            info=info.replace("\n", "\t\n"),
        )
        tools_add_child = [tree.add_child, ct.func_guide]
        func_dict = {func.__name__: func for func in tools_add_child}
        messages = [{"role": "user", "content": prompt}]
        check_start_prompt = "I see the help add child guidance"
        guidance = get_prompt("extra_help_add_child_prompt.txt").format(
            check_start_prompt=check_start_prompt
        )
        extra_guide_tool_call = get_func_tool_call(
            func_name="func_guide", result=guidance, guidance="Will Return Guidance"
        )
        while True:
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "LLM",
                            "content": "Query with Tools",
                            "level_delta": 1,
                        },
                    }
                )
            # print(f"[LLM] Query with Tools Start")
            # formatted_prompt = prompt.replace("\n", "\n    ")
            # print(f"  [INPUT]\n    {formatted_prompt}")
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "INPUT",
                            "content": prompt,
                            "level_delta": 0,
                        },
                    }
                )
            text, tool_calls = ct.llm.query_messages_with_tools(
                messages + extra_guide_tool_call,
                tools=tools_add_child,
                verbose=verbose,
            )
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "",
                            "content": "",
                            "level_delta": -1,
                        },
                    }
                )
            if text.startswith(check_start_prompt) == False:
                print(f"\nCheck Text Start Failed, Generate Again!\n")
                continue
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "SAMPLE",
                            "content": "Sampling Next Steps",
                            "level_delta": 1,
                        },
                    }
                )
            # print("\n[SAMPLE] Sampling Next Steps")
            sample_next_step = ["Resample"]
            for call in tool_calls:
                if ct.llm.format == "ollama":
                    call = call["function"]
                func_name = call["name"]
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args)
                if verbose:
                    formatted_args = ct.llm._format_arguments_for_display(
                        func_name, args
                    )
                    print(
                        f"\n  Sample will call function '{func_name}' with arguments:\n{formatted_args}"
                    )
                    sample_next_step.append(
                        f"\n  Sample will call function '{func_name}' with arguments:\n{formatted_args}"
                    )
            if human_decision:
                while True:
                    try:
                        # num = int(
                        #     input(
                        #         "Please choose one as next step (0 represents regenerate again): "
                        #     )
                        # )
                        next_option = ct.llm.ec.get_choice_response(
                            question="Please choose one as next step (0 represents regenerate again): ",
                            options=sample_next_step,
                        )
                        num = 0
                        for i in range(len(sample_next_step)):
                            if sample_next_step[i] == next_option:
                                num = i

                        if 0 <= num <= len(tool_calls):
                            break
                        else:
                            print(
                                f"❌ Invalid input. Please enter a number between 0 and {len(tool_calls)}."
                            )
                    except ValueError:
                        print("❌ Invalid input. Please enter a valid integer.")

                if num == 0:
                    if verbose:
                        ct.llm.ec.send_message(
                            {
                                "type": "info",
                                "data": {
                                    "category": "",
                                    "content": "",
                                    "level_delta": -1,
                                },
                            }
                        )
                    continue
                tool_calls = [tool_calls[num - 1]]
            else:
                guidance = (
                    f"Below are the sample tool calls\n"
                    f"{json.dumps(tool_calls, ensure_ascii=False, indent=2)}\n"
                    "You must choose only **ONE** as your next step tool call."
                )
                choose_add_child_tool_call = get_func_tool_call(
                    func_name="func_guide",
                    result="Will Return Guidance",
                    guidance=guidance,
                )
                if verbose:
                    ct.llm.ec.send_message(
                        {
                            "type": "info",
                            "data": {
                                "category": "LLM",
                                "content": "Select Automatically",
                                "level_delta": 1,
                            },
                        }
                    )
                # print(f"[LLM] Query with Tools Start")
                # formatted_prompt = prompt.replace("\n", "\n    ")
                # print(f"  [INPUT]\n    {formatted_prompt}")
                if verbose:
                    ct.llm.ec.send_message(
                        {
                            "type": "info",
                            "data": {
                                "category": "INPUT",
                                "content": prompt,
                                "level_delta": 0,
                            },
                        }
                    )
                text, tool_calls = ct.llm.query_messages_with_tools(
                    messages + tc.get_value() + choose_add_child_tool_call,
                    tools=tools_add_child,
                    verbose=verbose,
                )
                if verbose:
                    ct.llm.ec.send_message(
                        {
                            "type": "info",
                            "data": {
                                "category": "",
                                "content": "",
                                "level_delta": -1,
                            },
                        }
                    )
            if verbose:
                ct.llm.ec.send_message(
                    {
                        "type": "info",
                        "data": {
                            "category": "",
                            "content": "",
                            "level_delta": -1,
                        },
                    }
                )
            new_tool_calls = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                }
            ]

            for call in tool_calls:
                if ct.llm.format == "ollama":
                    call = call["function"]
                func_name = call["name"]
                args = call["function"]["arguments"]

                if func_name in func_dict:
                    if isinstance(args, str):
                        args = json.loads(args)
                    if verbose:
                        formatted_args = ct.llm._format_arguments_for_display(
                            func_name, args
                        )
                        print(
                            f"\n[CALL] function '{func_name}' with arguments:\n{formatted_args}"
                        )
                    result = func_dict[func_name](**args)
                else:
                    result = f"Function {func_name} not found"

                new_tool_calls.append(
                    {
                        "role": "tool",
                        "name": func_name,
                        "tool_call_id": call.get("id", ""),
                        "content": str(result),
                    }
                )
            tc.extend(new_tool_calls)
            if verbose:
                print(f"\n[FUNC] Complete handling add child.\n")
            break
        return "Successfully run help_add_child!"

    def help_return_to_parent(info: str):
        """
        Decide whether to return to the parent node or add a new child node
        based on the given information.

        This function helps manage the logical tree by using the accumulated
        information in `info`. It will call an LLM to determine whether the
        current task (node) is already completed. If the task is complete,
        it calls `return_to_parent`. Otherwise, it calls `help_add_child` to
        add a new subtask (child node).

        Args:
            info (str): Relevant information accumulated so far to guide
                        the decision of whether to return to parent or add
                        a new child task.
        """
        if verbose:
            print(f"\n[FUNC] Start handling return to parent.\n")
        info = ct.func_extract_factual_info(info)

        tree_json = tree.to_json()
        current_node = tree.current_node
        tool_calls = []
        if human_decision:
            decision = input(
                "If you think the task is completed, please input 'Yes', otherwise input your guidance"
            )
            if decision == "Yes":
                prompt = f"Human think the below info support back:\n{info}"
                tools_return_to_parent = [tree.return_to_parent]
                tool_calls.extend(
                    get_func_tool_call(
                        func_name="return_to_parent", text=prompt, success=True
                    )
                )
            else:
                guidance = f"Human give the following guidance:{decision}\n"
                tools_return_to_parent = [ct.func_guide]
                tool_calls.extend(
                    get_func_tool_call(
                        func_name="func_guide",
                        result=None,
                        guidance=guidance,
                    )
                )
        else:
            if current_node.back_num >= 0:
                prompt = get_prompt("help_return_to_parent_resample_prompt.txt").format(
                    tree_json=tree_json,
                    current_node_name=current_node.name,
                    current_node_description=current_node.description,
                    info=info.replace("\n", "\t\n"),
                )
                tools_return_to_parent = [help_add_child]
            else:
                prompt = get_prompt("help_return_to_parent_prompt.txt").format(
                    tree_json=tree_json,
                    current_node_name=current_node.name,
                    current_node_description=current_node.description,
                    info=info.replace("\n", "\t\n"),
                )
                tools_return_to_parent = [tree.return_to_parent, ct.func_guide]
            current_node.back_num -= 1
        func_dict = {func.__name__: func for func in tools_return_to_parent}
        if len(tool_calls) == 0:
            messages = [{"role": "user", "content": prompt}]
            print(f"[LLM] Query with Tools Start")
            formatted_prompt = prompt.replace("\n", "\n    ")
            print(f"  [INPUT]\n    {formatted_prompt}")
            text, tool_calls = ct.llm.query_messages_with_tools(
                messages,
                tools=tools_return_to_parent,
                verbose=verbose,
            )
        new_tool_calls = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            }
        ]
        for call in tool_calls:
            if ct.llm.format == "ollama":
                call = call["function"]
            func_name = call["name"]
            args = call["function"]["arguments"]

            if func_name in func_dict:
                if isinstance(args, str):
                    args = json.loads(args)
                if verbose:
                    formatted_args = ct.llm._format_arguments_for_display(
                        func_name, args
                    )
                    print(
                        f"\n[CALL] function '{func_name}' with arguments:\n{formatted_args}"
                    )
                result = func_dict[func_name](**args)
            else:
                result = f"Function {func_name} not found"

            new_tool_calls.append(
                {
                    "role": "tool",
                    "name": func_name,
                    "tool_call_id": call.get("id", ""),
                    "content": str(result),
                }
            )
        tc.extend(new_tool_calls)
        if verbose:
            print(f"\n[FUNC] Complete handling return to parent.\n")
        return "Successfully run help_return_to_parent!"

    def UpdataFunc():
        tool_calls = tc.get_value()
        if len(json.dumps(tool_calls)) > tc.MAX_CHAR:
            id = min(15, int(len(tool_calls) * 2 / 3))
            summary_tool_calls = query_summary_for_tool_calls(
                ct.llm, tc, id, tc.MAX_CHAR / 5
            )
            tc.insert_summarize_tool_calls(summary_tool_calls, id)

    # tc.UpdataFunc = UpdataFunc
    tc.mode = "Simple"
    if tools is None:
        tools = [
            ct.func_cat,
            ct.func_cat_with_llm,
            ct.func_write,
            ct.func_modify,
            ct.func_insert,
            ct.func_append,
            ct.func_prepend,
            ct.func_python,
            ct.func_cmd,
            ct.func_matlab,
            ct.func_fetch_info_from_pmc_with_llm,
            ct.func_human,
            ct.func_guide,
            # ct.func_think,
            ct.add_child,
            ct.return_to_parent,
            help_add_child,
            help_return_to_parent,
        ]
    # else:
    #     tools.extend(
    #         [
    #             help_add_child,
    #             help_return_to_parent,
    #         ]
    #     )
    write_tools = [ct.func_write]

    for idx, (goal_name, goal_description) in enumerate(goal_list, start=1):
        result = tree.add_child(goal_name, goal_description)
        tc.extend(
            get_func_tool_call(
                "add_child",
                result=result,
                goal_name=goal_name,
                goal_description=goal_description,
            )
        )

        if verbose:
            print(f"\n[INFO] Starting LLM query for goal {idx}...\n")

        # guidance = get_prompt("tools_emphasize_prompt.txt").format(
        #     tool_descriptions=summarize_tools(tools).replace("\n", "\t\n"),
        # )
        # extra_guide_tool_call = get_func_tool_call(
        #     func_name="func_guide", result=guidance, guidance="Will Return Guidance"
        # )
        extra_guide_tool_call = []
        guidance = get_prompt("extra_guidance_prompt.txt").format(
            current_node_name=tree.current_node.name,
            current_node_description=tree.current_node.description,
        )
        extra_guide_tool_call.extend(
            get_func_tool_call(
                func_name="func_guide", result=guidance, guidance="Will Return Guidance"
            )
        )

        # text = ct.llm.query_with_tools_by_attention(
        #     prompt=lambda: generate_prompt_for_solve_goal_list(
        #         goal_description, tree, tree_filepath
        #     ),
        #     max_steps=max_steps,
        #     tools=tools,
        #     tc=tc,
        #     extra_guide_tool_call=extra_guide_tool_call,
        #     verbose=verbose,
        #     stop_condition=stop_condition,
        #     check_start_prompt="I see the guidance",
        #     human_check_before_calling=human_check_before_calling,
        # )
        
        text = ct.llm.query_with_local_memory(
            prompt=lambda: generate_prompt_for_solve_goal_list(
                goal_description, tree, tree_filepath
            ),
            max_steps=max_steps,
            tools=tools,
            write_tools=write_tools,
            tc=tc,
            local_memory_folder="tmp/local_memory",
            memory_unit_number=2,
            extra_guide_tool_call=extra_guide_tool_call,
            verbose=verbose,
            stop_condition=stop_condition,
            check_start_prompt="I see the guidance",
            human_check_before_calling=human_check_before_calling,
        )

        if verbose:
            print(f"\n[INFO] Finished processing goal {idx}\n")

def get_github_issue(sample: dict) -> dict:
    """
    给定 SWE-bench 样本，返回该问题的 GitHub issue 链接、
    对应仓库克隆地址和 base_commit（出错代码版本）。

    参数:
        sample: dict, 包含 'repo', 'instance_id', 'problem_statement', 'base_commit'
    返回:
        {
            "repo": "astropy/astropy",
            "repo_clone_url": "https://github.com/astropy/astropy.git",
            "issue_url": "https://github.com/astropy/astropy/issues/12907",
            "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
            "problem": "<issue 描述文本>"
        }
    """
    repo = sample["repo"]  # "astropy/astropy"
    instance_id = sample["instance_id"]  # "astropy__astropy-12907"
    base_commit = sample["base_commit"]  # 出错版本
    issue_num = instance_id.split("-")[-1]  # "12907"

    return {
        "repo": repo,
        "repo_clone_url": f"https://github.com/{repo}.git",
        "issue_url": f"https://github.com/{repo}/issues/{issue_num}",
        "base_commit": base_commit,
        "problem": sample.get("problem_statement", "").strip(),
    }


def fix_github_error(
    sample: dict,
    ct: custom_tools,
    max_steps: int,
    tc: Tool_Calls,
    package_name: str,
    verbose: bool,
    whether_recreate: bool,
    save_filepath: str,
):
    if whether_recreate:
        if verbose:
            ct.llm.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "ENVIRONMENT",
                        "content": "Creating Docker environment...",
                        "level_delta": 0,
                    },
                }
            )
            # print("\n[INFO] Creating Docker environment...\n")

        ct.func_create_env()
    issue = get_github_issue(sample)
    github_url = issue["repo_clone_url"]
    base_commit = issue["base_commit"]
    if verbose:
        ct.llm.ec.send_message(
            {
                "type": "info",
                "content": {
                    "category": "CLONE GITHUB",
                    "content": "Cloning GitHub repository...",
                    "level_delta": 0,
                },
            }
        )
    print("\n[INFO] Cloning GitHub repository...\n")

    ct.func_cmd(f"rm -rf ./*")
    print(github_url)
    result = ct.func_git_clone(github_url, package_name)

    new_tool_calls = get_func_tool_call(
        func_name="func_cmd",
        result=result,
        command=f"git clone {github_url} {package_name}",
    )
    tc.extend(new_tool_calls)
    if base_commit is not None:
        command = f"cd {package_name} && git checkout {base_commit} && pip install -e ."
        result = ct.func_cmd(command)

        new_tool_calls = get_func_tool_call(
            func_name="func_cmd",
            result=result,
            command=command,
        )
        tc.extend(new_tool_calls)
    tools = [
        ct.func_cat,
        # ct.func_cat_with_llm,
        ct.func_write,
        ct.func_modify,
        ct.func_insert,
        ct.func_append,
        ct.func_prepend,
        ct.func_python,
        ct.func_cmd,
        # ct.func_matlab,
        # ct.func_fetch_info_from_pmc_with_llm,
        # ct.func_human,
        ct.func_reflect,
        ct.func_guide,
        # ct.func_think,
        ct.add_child,
        ct.return_to_parent,
    ]
    goal_list = [
        (
            "fix github issue",
            get_prompt("github_issue_fix_prompt.txt").format(
                filepath=save_filepath, problem=issue["problem"].replace("\n", "\n\t"), base_commit=base_commit
            ),
        )
        # (
        #     "improve existing github patch",
        #     get_prompt("github_patch_improvement_prompt.txt").format(
        #         filepath=save_filepath,
        #         problem=issue["problem"].replace("\n", "\n\t"),
        #         base_commit=base_commit,
        #     ),
        # )
    ]

    return solve_goal_list(
        goal_list=goal_list,
        ct=ct,
        max_steps=max_steps,
        tc=tc,
        verbose=verbose,
        human_decision=False,
        tools=tools,
    )
    

def radiogenomics_agent(
    sample: dict,
    ct: custom_tools,
    max_steps: int,
    tc: Tool_Calls,
    package_name: str,
    verbose: bool,
    whether_recreate: bool,
    save_filepath: str,
):
    if whether_recreate:
        if verbose:
            ct.llm.ec.send_message(
                {
                    "type": "info",
                    "data": {
                        "category": "ENVIRONMENT",
                        "content": "Creating Docker environment...",
                        "level_delta": 0,
                    },
                }
            )
            # print("\n[INFO] Creating Docker environment...\n")

        ct.func_create_env()
    issue = get_github_issue(sample)
    github_url = issue["repo_clone_url"]
    base_commit = issue["base_commit"]
    if verbose:
        ct.llm.ec.send_message(
            {
                "type": "info",
                "content": {
                    "category": "CLONE GITHUB",
                    "content": "Cloning GitHub repository...",
                    "level_delta": 0,
                },
            }
        )
    print("\n[INFO] Cloning GitHub repository...\n")

    ct.func_cmd(f"rm -rf ./*")
    print(github_url)
    result = ct.func_git_clone(github_url, package_name)

    new_tool_calls = get_func_tool_call(
        func_name="func_cmd",
        result=result,
        command=f"git clone {github_url} {package_name}",
    )
    tc.extend(new_tool_calls)
    if base_commit is not None:
        command = f"cd {package_name} && git checkout {base_commit}"
        result = ct.func_cmd(command)

        new_tool_calls = get_func_tool_call(
            func_name="func_cmd",
            result=result,
            command=command,
        )
        tc.extend(new_tool_calls)
    tools = [
        ct.func_cat,
        # ct.func_cat_with_llm,
        ct.func_write,
        ct.func_modify,
        ct.func_insert,
        ct.func_append,
        ct.func_prepend,
        ct.func_python,
        ct.func_cmd,
        # ct.func_matlab,
        # ct.func_fetch_info_from_pmc_with_llm,
        # ct.func_human,
        ct.func_reflect,
        ct.func_guide,
        # ct.func_think,
        ct.add_child,
        ct.return_to_parent,
    ]
    goal_list = [
        (
            "fix github issue",
            get_prompt("github_issue_fix_prompt.txt").format(
                filepath=save_filepath, problem=issue["problem"].replace("\n", "\n\t"), base_commit=base_commit
            ),
        ),
        (
        "improve existing github patch",
        get_prompt("github_patch_improvement_prompt.txt").format(
            filepath=save_filepath,
            problem=issue["problem"].replace("\n", "\n\t"),
            base_commit=base_commit,
        ),
    )
    ]

    return solve_goal_list(
        goal_list=goal_list,
        ct=ct,
        max_steps=max_steps,
        tc=tc,
        verbose=verbose,
        human_decision=False,
        tools=tools,
    )


def query_based_on_tool_calls(
    request: str,
    ct: custom_tools,
    tc: Tool_Calls,
    verbose: bool,
):
    tree = LogicalTree("goal", request)
    tools = [
        ct.func_cat,
        ct.func_write,
        ct.func_modify,
        ct.func_insert,
        ct.func_append,
        ct.func_prepend,
        ct.func_python,
        ct.func_cmd,
        ct.func_matlab,
        ct.func_fetch_info_from_pmc_with_llm,
        ct.func_human,
        tree.add_child,
        tree.return_to_parent,
    ]
    text = ct.llm.query_with_tools_by_attention(
        prompt=get_prompt("query_based_on_tool_calls_prompt.txt").format(
            request=request.replace("\n", "\t\n")
        ),
        max_steps=1,
        tools=tools,
        tc=tc,
        verbose=verbose,
    )
    return text


def query_summary_for_tool_calls(llm: LLM, tc: Tool_Calls, id: int, max_char: int):
    print("\nCalling Summary\n")
    tool_calls = tc.get_value()
    tool_calls_need_summary = tool_calls[:id]
    tool_calls_not_need_summary = tool_calls[id:]
    prompt = get_prompt("query_summary_for_tool_calls_prompt.txt").format(
        part1=trans_tool_calls_to_str(tool_calls_need_summary).replace("\n", "\t\n"),
        part2=trans_tool_calls_to_str(tool_calls_not_need_summary).replace(
            "\n", "\t\n"
        ),
        max_char=max_char,
    )
    summary = llm.query(prompt=prompt, verbose=False)
    guidance = f"Previoud Tool Summary is {summary}"
    return get_func_tool_call(
        func_name="func_guide", result=guidance, guidance="Will Return Guidance"
    )


def trans_tool_calls_to_str(tool_calls_messages: List[Dict[str, Any]]) -> str:
    """
    将工具调用消息列表提取关键信息并转换为可读字符串。
    不使用 JSON 格式，而是生成简明描述。

    Args:
        tool_calls_messages: 工具调用消息列表

    Returns:
        一个字符串，包含所有关键信息
    """
    lines = []

    for msg in tool_calls_messages:
        role = msg.get("role", "unknown")

        # 如果消息里有 tool_calls 列表
        if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
            for call in msg["tool_calls"]:
                func_name = call.get("function", {}).get(
                    "name", call.get("name", "unknown")
                )
                arguments = call.get("function", {}).get("arguments", {})
                args_str = ", ".join(f"{k}={v}" for k, v in arguments.items())
                lines.append(f"[{role}] {func_name}({args_str})")
        else:
            # 普通 message
            content = msg.get("content", "")
            lines.append(f"[{role}] {content}")

    # 合并成一个字符串，每条记录换行
    return "\n".join(lines)
