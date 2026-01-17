from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.services.agents import get_func_tool_call


def sanitize_package_name(raw_name: str) -> str:
    """
    Convert an arbitrary SWE-bench instance identifier into a filesystem-friendly package name.
    Only keep alphanumerics, dash, underscore, and dot; collapse other runs into a single underscore.
    """
    if not raw_name:
        return "swe_bench_repo"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "swe_bench_repo"


@dataclass
class SWEInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    tests: List[str]
    test_patch: Optional[str] = None
    hints_text: Optional[Iterable[str]] = None
    additional_context: Optional[str] = None

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{self.repo}.git"

    @property
    def safe_package_name(self) -> str:
        return sanitize_package_name(self.instance_id)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SWEInstance":
        return cls(
            instance_id=str(payload.get("instance_id", "")),
            repo=str(payload.get("repo", "")),
            base_commit=str(payload.get("base_commit", "")),
            problem_statement=str(payload.get("problem_statement", "")),
            tests=list(payload.get("tests", [])),
            test_patch=payload.get("test_patch"),
            hints_text=payload.get("hints_text"),
            additional_context=payload.get("additional_context"),
        )


def build_goal_list_from_instance(instance: SWEInstance) -> List[Tuple[str, str]]:
    """
    Build a single-goal list compatible with build_agent using the SWE-bench instance information.
    """
    hints_block = ""
    if instance.hints_text:
        hints = "\n".join(f"- {hint}" for hint in instance.hints_text if hint)
        if hints:
            hints_block = f"\nHints provided:\n{hints}\n"

    tests_block = ""
    if instance.tests:
        tests_list = "\n".join(f"- {command}" for command in instance.tests)
        tests_block = (
            "\nVerification tests (must pass without relying on provided patch):\n"
            f"{tests_list}\n"
        )

    extra_block = (
        f"\nAdditional context from dataset:\n{instance.additional_context}\n"
        if instance.additional_context
        else ""
    )

    description = dedent(
        f"""
        SWE-bench instance: {instance.instance_id}
        Repository: {instance.repo}
        Base commit: {instance.base_commit}

        Problem description:
        {instance.problem_statement.strip()}
        """
    ).strip()

    goal_description = "\n".join(
        block for block in [description, hints_block.strip(), tests_block.strip(), extra_block.strip()] if block
    )

    goal_name = f"SWE-bench task {instance.instance_id or instance.repo}"
    return [(goal_name, goal_description)]


def post_clone_setup_tool_calls(
    ct,
    instance: SWEInstance,
    package_name: str,
) -> List[Dict[str, Any]]:
    """
    After cloning the repository, ensure it matches the SWE-bench environment by
    checking out the base commit and applying the dataset's test patch if provided.

    Returns a list of tool-call like records so callers can extend their Tool_Calls tracker.
    """
    tool_messages: List[Dict[str, Any]] = []

    checkout_result = ct.func_git_checkout(package_name=package_name, commit=instance.base_commit)
    tool_messages.extend(
        get_func_tool_call(
            func_name="func_git_checkout",
            result=checkout_result,
            package_name=package_name,
            commit=instance.base_commit,
        )
    )

    if instance.test_patch:
        apply_result = ct.func_git_apply_patch(package_name=package_name, patch_text=instance.test_patch)
        tool_messages.extend(
            get_func_tool_call(
                func_name="func_git_apply_patch",
                result=apply_result,
                package_name=package_name,
                has_patch=True,
            )
        )

    return tool_messages


def ensure_absolute_repo_path(main_dir: str, package_name: str) -> Path:
    return Path(main_dir) / package_name
