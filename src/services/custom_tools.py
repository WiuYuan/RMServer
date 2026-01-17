# src/services/custom_tools.py

import glob
from typing import Callable, Dict, List, Any
from src.services.llm import LLM
import re
import os
import subprocess
from src.utils.prompts import get_prompt
from src.services.pubmed_scraper import (
    fetch_pmc_article,
    remove_references,
    extract_main_article_body,
)
from src.services.agents import Tool_Calls, get_func_tool_call
import ast
from youtube_transcript_api import YouTubeTranscriptApi
import requests, time, random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json


class custom_tools:
    def __init__(
        self,
        HOST_DIR: str = "",
        HOST_DATA_DIR: str = "",
        WRITE_DIR: str = "",
        MATLAB_PATH: str = "matlab",
        ENV_NAME: str = "",
        LOCAL_TMP_PATH: str = "",
        BASE_ENV: str = "",
        PMC_URL: str = "",
        MAIN_DIR: str = "/workspace",
        MAIN_DATA_DIR: str = "/data",
        PYTHON_PATH: str = "python",
        REMOTE_TMP_PATH: str = "/tmp",
        CHUNK_SIZE: int = 5000,
        llm: LLM = LLM(),
        verbose: bool = True,
    ):
        self.HOST_DIR = HOST_DIR
        self.HOST_DATA_DIR = HOST_DATA_DIR
        self.WRITE_DIR = WRITE_DIR
        self.MAIN_DIR = MAIN_DIR
        self.MAIN_DATA_DIR = MAIN_DATA_DIR
        self.PYTHON_PATH = PYTHON_PATH
        self.MATLAB_PATH = MATLAB_PATH
        self.ENV_NAME = ENV_NAME
        self.LOCAL_TMP_PATH = LOCAL_TMP_PATH
        self.REMOTE_TMP_PATH = REMOTE_TMP_PATH
        self.BASE_ENV = BASE_ENV
        self.PMC_URL = PMC_URL
        self.CHUNK_SIZE=CHUNK_SIZE
        self.llm = llm
        self.verbose = verbose

    def func_error(self, error: str) -> str:
        prompt = get_prompt("func_error_prompt.txt").format(
            error_message=error, MAIN_DIR=self.MAIN_DIR
        )
        return self.llm.query(prompt)

    def func_local_cmd(self, command: str) -> str:
        """
        This function writes the command to a .sh file and then executes it using bash in HOST.

        Args:
            command (str): The command to run.

        Returns:
            str: Captured stdout or simplified error message if failed.
        """
        # Split the command safely for subprocess (avoid bash parsing issues)
        local_path = self._write_to_local_tmp("_func_cmd", command)
        run_cmd = f"bash {local_path}"
        result = subprocess.run(run_cmd, shell=True, capture_output=True)
        try:
            output = result.stdout.decode("utf-8")
        except UnicodeDecodeError:
            return f"Error: Command output is not UTF-8 decodable."

        try:
            error = result.stderr.decode("utf-8")
        except UnicodeDecodeError:
            error = "Error: STDERR output is not UTF-8 decodable."

        if result.returncode == 0:
            return output
        else:
            error = self.func_error(error)
            return f"Error (code {result.returncode}):\nSTDOUT:\n{output}\nSTDERR:\n{error}"

    def _func_cmd(self, command: str) -> str:
        """
        This function writes the command to a .sh file and then executes it using bash, capture stdout/stderr, and optionally return selected chunks of stdout.

        Args:
            command (str): The command to run.
            return_chunk_indices (list[int]): 1-based indices of chunks to return.
                Negative indices are allowed (like Python lists). Default is [1, -1] (first and last chunk).

        Returns:
            str: Selected stdout chunks with metadata and simplified error message if failed.
        """
        # 将命令写入临时文件并执行
        if len(self.ENV_NAME) != 0:
            container_path = self._write_to_remote_tmp("_func_cmd", command)
            run_cmd = f"docker exec {self.ENV_NAME} bash -c 'cd {self.MAIN_DIR} && bash {container_path}'"
            result = subprocess.run(run_cmd, shell=True, capture_output=True)
        else:
            local_path = self._write_to_local_tmp("_func_cmd", command)
            run_cmd = f"bash -c 'cd {self.MAIN_DIR} && bash {local_path}'"
            result = subprocess.run(run_cmd, shell=True, capture_output=True)

        try:
            output = result.stdout.decode("utf-8")
            output = clean_training_logs(output)
        except UnicodeDecodeError:
            return "Error: Command output is not UTF-8 decodable."

        return output

    def func_cmd(self, command: str, return_chunk_indices: list[int] = [1, -1], max_time: float = 60.0) -> str:
        """
        This function writes the command to a .sh file and then executes it using bash, capture stdout/stderr, and optionally return selected chunks of stdout.

        Args:
            command (str): The command to run.
            return_chunk_indices (list[int]): 1-based indices of chunks to return.
                Negative indices are allowed (like Python lists). Default is [1, -1] (first and last chunk).
            max_time (float): Maximum execution time (seconds). Default 60.0 seconds.

        Returns:
            str: Selected stdout chunks with metadata and simplified error message if failed.
        """
        if isinstance(return_chunk_indices, str):
            # 尝试解析成列表
            try:
                return_chunk_indices = ast.literal_eval(return_chunk_indices)
                if not isinstance(return_chunk_indices, list):
                    return_chunk_indices = [1, -1]  # 默认值
            except Exception:
                return_chunk_indices = [1, -1]  # 默认值
        return_chunk_indices = [int(x) for x in return_chunk_indices]

        # 固定 chunk 设置
        CHUNK_SIZE = self.CHUNK_SIZE
        CHUNK_OVERLAP = int(CHUNK_SIZE / 10)

        # 将命令写入临时文件并执行
        if len(self.ENV_NAME) != 0:
            container_path = self._write_to_remote_tmp("_func_cmd", command)
            run_cmd = f"docker exec {self.ENV_NAME} bash -c 'cd {self.MAIN_DIR} && bash {container_path}'"
        else:
            local_path = self._write_to_local_tmp("_func_cmd", command)
            run_cmd = f"bash -c 'cd {self.MAIN_DIR} && bash {local_path}'"
        try:
            if max_time is None:
                # No timeout → do not pass the timeout parameter at all
                result = subprocess.run(
                    run_cmd,
                    shell=True,
                    capture_output=True,
                    env=os.environ,
                )
            else:
                result = subprocess.run(
                    run_cmd,
                    shell=True,
                    capture_output=True,
                    env=os.environ,
                    timeout=max_time,
                )
        except subprocess.TimeoutExpired as e:
            result = subprocess.CompletedProcess(
                args=e.cmd,
                returncode=-1,
                stdout=e.output or b"",
                stderr=(e.stderr or b"") + f"\n[Process timed out after {max_time}s]".encode(),
            )
        try:
            output = result.stdout.decode("utf-8")
            output = clean_training_logs(output)
        except UnicodeDecodeError:
            return "Error: Command output is not UTF-8 decodable."

        # 拆分 stdout 为 chunk
        output_chunks = chunk_text_by_chars(
            output, max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP
        )
        n_chunks = len(output_chunks)

        # 处理返回的 chunk indices，并去重
        seen = set()  # 保存已添加的 chunk 序号
        selected_chunks = []
        for idx in return_chunk_indices:
            if idx > n_chunks:
                idx = n_chunks
            if idx < -n_chunks:
                idx = -n_chunks
            py_idx = idx - 1 if idx > 0 else n_chunks + idx
            one_based_idx = py_idx + 1
            if one_based_idx not in seen:
                selected_chunks.append((one_based_idx, output_chunks[py_idx]))
                seen.add(one_based_idx)

        try:
            error = result.stderr.decode("utf-8")
        except UnicodeDecodeError:
            error = "Error: STDERR output is not UTF-8 decodable."
            
        MAX_STDERR_CHUNK = int(self.CHUNK_SIZE / 2)
            
        if len(error) > MAX_STDERR_CHUNK * 2:
            head = error[:MAX_STDERR_CHUNK]
            tail = error[-MAX_STDERR_CHUNK:]
            error = (
                f"{head}\n\n...[truncated middle {len(error) - 2*MAX_STDERR_CHUNK} chars]...\n\n{tail}"
            )

        status = "Run the Command Successfully"
        if result.returncode != 0:
            status = "Run the Command Failed"
            # error = self.func_error(error)

        # 构建输出说明
        metadata_lines = [
            f"Total chunks: {n_chunks}",
        ]
        chunk_texts = []
        for idx, chunk in selected_chunks:
            chunk_texts.append(f"--- Chunk {idx} ---\n{chunk}")

        return (
            f"(code {result.returncode}, status={status}):\n"
            + "\n".join(metadata_lines)
            + "\nSTDOUT (selected chunks):\n"
            + "\n...\n".join(chunk_texts)
            + f"\nSTDERR:\n{error}"
        )

    def func_ls(self, filepath: str) -> str:
        """
        List the contents of a directory or show a file at a specified relative path.

        Parameters:
            filepath (str): Relative path under MAIN_DIR, e.g., 'agent1/main.sh'.

        Returns:
            str: Contents of the directory or the filename if it's a file, or an error message if the path doesn't exist.

        Example:
            input: 'agent1'
            output: list of files/folders under f'{MAIN_DIR}/agent1'
        """
        base_path = self.MAIN_DIR
        full_path = os.path.join(base_path, filepath)
        command = f"cd {full_path} && ls --color=never"
        return self.func_cmd(command)

    def _func_cat(self, filepath: str) -> str:
        """
        Safely display contents of a file under MAIN_DIR, only if UTF-8.

        Parameters:
            filepath (str): Relative path under MAIN_DIR, e.g., 'agent1/main.sh'.

        Returns:
            str: cat output.
        """
        full_path = os.path.join(self.MAIN_DIR, filepath)

        # 使用 cat 读取整个文件，并直接让 func_cmd 处理 chunk
        command = f"cat {full_path}"

        # 调用 func_cmd 并传递 chunk 索引
        return self._func_cmd(command)

    def func_cat(self, filepath: str, return_chunk_indices: list[int] = [1, -1]) -> str:
        """
        Safely display contents of a file under MAIN_DIR, only if UTF-8,
        and optionally return selected chunks of the output.

        Parameters:
            filepath (str): Relative path under MAIN_DIR, e.g., 'agent1/main.sh'.
            return_chunk_indices (list[int]): 1-based indices of chunks to return. Negative indices allowed. Default [1, -1].
            ⚠️ Important:
                - These are **chunk indices**, not line numbers.
                - Do NOT pass raw line numbers (such as those returned by grep -n),
                  because the file is divided into a small number of chunks (e.g., Total chunks: 3),
                  not hundreds of lines.
        Returns:
            str: Selected stdout chunks with metadata and simplified error message if failed.
        """
        full_path = os.path.join(self.MAIN_DIR, filepath)

        # 使用 cat 读取整个文件，并直接让 func_cmd 处理 chunk
        command = f"cat {full_path}"

        # 调用 func_cmd 并传递 chunk 索引
        return self.func_cmd(command, return_chunk_indices=return_chunk_indices)

    def _func_write(self, filepath: str, text: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return filepath

    def _write_to_local_tmp(self, tmp_filename: str, text: str) -> str:
        os.makedirs(self.LOCAL_TMP_PATH, exist_ok=True)
        tmp_filepath = os.path.join(self.LOCAL_TMP_PATH, tmp_filename)
        with open(tmp_filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return tmp_filepath

    def transfer_file_to_docker(self, local_path: str, container_path: str) -> str:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        container_dir = os.path.dirname(container_path)
        mkdir_cmd = f"docker exec {self.ENV_NAME} mkdir -p {container_dir}"
        subprocess.run(mkdir_cmd, shell=True, capture_output=True, text=True)
        cp_cmd = ["docker", "cp", local_path, f"{self.ENV_NAME}:{container_path}"]
        subprocess.run(cp_cmd, capture_output=True, text=True)
        return

    def func_write(self, filepath: str, text: str):
        """
        Write the given text to a file inside the project directory.

        Parameters:
            filepath (str): Relative path under MAIN_DIR, e.g., 'agent1/main.sh'.
            text (str): Text content to write to the file.

        Returns:
            str: Confirmation message indicating the file has been written.

        Example:
            input: 'agent1/main.sh', 'echo hello'
            output: 'Written to agent1/main.sh'
        """
        if len(self.WRITE_DIR) > 0:
            abs_target = os.path.abspath(os.path.join(self.MAIN_DIR, filepath))
            abs_write_dir = os.path.abspath(self.WRITE_DIR)

            # ✅ 检查是否在允许目录内
            if not abs_target.startswith(abs_write_dir + os.sep):
                return f"[Error] Unauthorized write attempt to {abs_target}. Allowed directory: {abs_write_dir}"
        if len(self.ENV_NAME) == 0:
            local_filepath = os.path.join(self.MAIN_DIR, filepath)
            self._func_write(filepath=local_filepath, text=text)
        else:
            tmp_filename = "_func_write_tmp_file"
            tmp_filepath = self._write_to_local_tmp(tmp_filename, text)
            container_path = os.path.join(self.MAIN_DIR, filepath)
            self.transfer_file_to_docker(tmp_filepath, container_path)
        return f"Written to {filepath}"

    def _write_to_remote_tmp(self, tmp_filename: str, text: str) -> str:
        container_path = os.path.join(self.REMOTE_TMP_PATH, tmp_filename)
        os.makedirs(self.REMOTE_TMP_PATH, exist_ok=True)
        self.func_write(container_path, text)
        return container_path

    def func_python(self, filepath: str, workdir: str = "."):
        """
        Run a Python script located at 'filepath' inside the given working directory
        (relative to MAIN_DIR).

        The function will first 'cd ${MAIN_DIR}/{workdir}' and then run the script
        using the designated Python environment.

        Any error messages generated by the script will be captured and automatically
        simplified using the LLM before being returned.

        Example:
        func_python(filepath="test/test.py", workdir="agent1")
        will run:
        'cd ${MAIN_DIR}/agent1 && python ${MAIN_DIR}/agent1/test/test.py'
        """
        abs_workdir = os.path.join(self.MAIN_DIR, workdir)
        full_path = os.path.join(self.MAIN_DIR, filepath)
        command = f"cd {abs_workdir} && {self.PYTHON_PATH} {full_path}"
        return self.func_cmd(command)

    def func_pip_install(self, package: str):
        """
        Install a Python package.

        The function will run:
        'pip install {package}'

        Any error messages generated will be captured and simplified using LLM.

        Example:
        func_pip_install(package="numpy")
        """
        command = f"{self.PYTHON_PATH} -m pip install {package}"
        return self.func_view_text(
            self.func_cmd(command), "Simplify pip command outputs"
        )

    def func_pip_uninstall(self, package: str):
        """
        Uninstall a Python package.

        The function will run:
        'pip uninstall -y {package}'

        Any error messages generated will be captured and simplified using LLM.

        Example:
        func_pip_uninstall(package="numpy")
        """
        command = f"{self.PYTHON_PATH} -m pip uninstall {package}"
        return self.func_view_text(
            self.func_cmd(command), "Simplify pip command outputs"
        )

    def func_matlab(self, filepath: str):
        """
        Run a MATLAB script at the specified path.

        Note:
        - You must use this function to run MATLAB scripts instead of func_cmd.
        - This function is configured with a dedicated MATLAB module, so it will always run successfully
        even if MATLAB is not in the system PATH.

        Example:
            input 'test/test.m' will run the script.
            MATLAB will first change directory to the script's folder,
            then execute the script.
        """
        full_path = os.path.join(self.HOST_DIR, filepath)
        dir_path = os.path.dirname(full_path)
        script_name = os.path.splitext(os.path.basename(full_path))[0]

        command = command = (
            f"{self.MATLAB_PATH} -batch \"cd('{dir_path}'); {script_name}\""
        )
        return self.func_local_cmd(command)

    def func_human(self, error: str):
        """
        Use this tool when the model cannot resolve an issue automatically.

        This agent is specifically designed to request **human assistance** for errors,
        unresolved problems, or ambiguous situations. When invoked, it will prompt a human
        to provide a solution or guidance and will return the human-provided response.

        The model should call this tool **only when all automated methods have failed**
        and a human response is necessary to continue.
        """
        print(f"\nError occurred:\n{error}\n")
        llm_answer = self.llm.query(
            get_prompt("func_human_prompt.txt").format(request=error)
        )

        if llm_answer[:2] == "No":
            result = input("Please provide your solution: ")
        else:
            result = llm_answer
        return result

    def func_guide(self, guidance: str):
        """
        Never use this tool by yourself.

        This agent is specifically designed to derive someone else guidance.

        it will return guidance by others to you.
        """
        return guidance

    def func_view_with_llm(self, text: str, request):
        """
        Let another LLM view and analyze the content of a specified file.
        - text: the file text
        - request: a question or instruction regarding the file content
        The tool will read the file content and, together with the provided instruction, return an AI analysis or answer based on the file.
        """
        prompt = get_prompt("func_view_prompt.txt").format(request=request, file=text)
        return self.llm.query(prompt)

    def func_cat_with_llm(
        self, filepath: str, request: str, max_chars: int = 5000, overlap: int = 500
    ) -> str:
        """
        Let another LLM view and analyze the content of a specified file in chunks.

        Procedure:
            1. Read the full file content using _func_cat.
            2. Split the content into chunks based on max_chars with overlap.
            3. For each chunk, call the LLM to analyze the chunk with the provided request.
            4. Combine the results and return as a single string.

        Args:
            filepath (str): The path to the file to read.
            request (str): A question or instruction regarding the file content.
            max_chars (int): Maximum characters per chunk.
            overlap (int): Number of overlapping characters between chunks.

        Returns:
            str: Combined LLM analysis of all chunks.
        """
        # 1. 获取文件完整内容
        full_text = self._func_cat(filepath)

        # 2. 按字符拆分成 chunks
        chunks = chunk_text_by_chars(full_text, max_chars=max_chars, overlap=overlap)

        # 3. 对每个 chunk 用 LLM 根据 request 得到分析
        results = []
        for i, chunk in enumerate(chunks):
            prompt = get_prompt("func_view_prompt.txt").format(
                request=request, file=chunk
            )
            result_chunk = self.llm.query(prompt)
            results.append(f"--- Chunk {i+1} ---\n{result_chunk}")

        # 4. 合并结果
        return "\n".join(results)

    def func_generate_code(self, filepath: str, request: str):
        """
        Generate complete, runnable code strictly based on the given request and write it to the specified file.

        Args:
            filepath (str): Full path to the file to create or update.
            request (str): Code specification containing all necessary details such as data locations, file paths, dependencies, and expected behavior.

        Notes:
            - The AI has no prior knowledge of your environment or files. All information required to generate correct code must be included in the request.
            - The output must be fully functional code, without explanations, comments, or extra text.
            - If the request cannot be fulfilled, return 'NOT_FOUND'.
        """
        prompt = get_prompt("func_generate_code_prompt.txt").format(request=request)
        code = self.llm.query(prompt)
        return f"Code generate completedly. {self.func_write(filepath, code)}"

    def func_create_env(self, platform: str = "linux/arm64"):
        self.func_local_cmd(f"rm -rf {self.HOST_DIR}/* {self.HOST_DATA_DIR}/*")
        self.func_remove_env()
        self.func_local_cmd(f"mkdir -p {self.HOST_DIR} {self.HOST_DATA_DIR}")
        return self.func_local_cmd(
            f"docker run -d "
            f"--name {self.ENV_NAME} "
            f"-v {self.HOST_DIR}:{self.MAIN_DIR} "
            f"-v {self.HOST_DATA_DIR}:{self.MAIN_DATA_DIR} "
            f"--platform {platform} "
            f"{self.BASE_ENV} tail -f /dev/null"
        )

    def func_remove_env(self):
        return self.func_local_cmd(f"docker rm -f {self.ENV_NAME} || true")

    def func_git_clone(self, url: str, package_name: str):
        cmd = (
            f"cd {self.MAIN_DIR} && "
            f"until git clone {url} {package_name}; do "
            f"echo '⚠️  Git clone failed, retrying in 10s ...'; "
            f"sleep 10; "
            f"done"
        )
        return self.func_cmd(cmd, max_time=None)
        # return self.func_cmd(f"cd {self.MAIN_DIR} && git clone {url} {package_name}")

    def func_git_checkout(self, package_name: str, commit: str):
        repo_dir = os.path.join(self.MAIN_DIR, package_name)
        command = f"cd {repo_dir} && git checkout {commit}"
        return self.func_cmd(command)

    def func_git_apply_patch(self, package_name: str, patch_text: str):
        """
        Apply a diff patch inside the cloned repository, storing the patch in a temporary file first.
        """
        if not patch_text.endswith("\n"):
            patch_text = patch_text + "\n"
        patch_path = self._write_to_remote_tmp("_func_git_apply_patch.diff", patch_text)
        repo_dir = os.path.join(self.MAIN_DIR, package_name)
        command = f"cd {repo_dir} && git apply --whitespace=fix {patch_path}"
        return self.func_cmd(command)

    def func_fetch_info_from_pmc_with_llm(self, request):
        """
        Fetch information from a PMC article using LLM.

        Args:
            request (str): A question or instruction regarding the content of the article.

        Returns:
            str: Analysis or answer from the LLM based on the article content.
        """

        """
        Fetch information from the PMC article corresponding to the github package using LLM.

        Args:
            request (str): A question or instruction regarding the content of the article.

        Returns:
            str: Analysis or answer from the LLM based on the article content.
        """

        text = remove_references(
            extract_main_article_body(fetch_pmc_article(self.PMC_URL))
        )
        return self.func_view_with_llm(text, request)

    def func_modify(self, filepath: str, old_text: str, new_text: str) -> str:
        """
        Safely attempt to modify existing text in a file by replacing old_text with new_text. If the pattern is not found, an error message will be returned. After modifying, it's advisable to double-check the file for proper format.
            filepath (str): Relative path under MAIN_DIR, e.g., 'agent1/main.sh'.
            old_text (str): The text pattern to be replaced.
            new_text (str): The new text to replace the old text.

        Returns:
            str: Confirmation message indicating the modification has been done.

        Example:
            input: 'agent1/main.sh', 'echo hello', 'echo world'
            output: 'Modified agent1/main.sh: replaced "echo hello" with "echo world"'
        """
        # First read the current file content
        current_content = self._func_cat(filepath)

        # Check if old_text exists in the file
        if old_text not in current_content:
            return f"Error: Text '{old_text}' not found in {filepath}"

        # Replace the text
        modified_content = current_content.replace(old_text, new_text)

        # Write the modified content back to the file
        return self.func_write(filepath, modified_content)

    def func_insert(self, filepath: str, insert_after: str, new_text: str) -> str:
        """
        Safely attempt to insert new text after a specific pattern in a file. If the pattern is not found, an error message will be returned. After inserting, it's advisable to double-check the file for proper format. If the pattern is not found, an error message will be returned. After modifying, it's advisable to double-check the file for proper format.
            filepath (str): Relative path under MAIN_DIR.
            insert_after (str): The text pattern after which to insert new content.
            new_text (str): The text to insert.

        Returns:
            str: Confirmation message.

        Example:
            input: 'agent1/main.sh', 'import os', 'import sys'
            output: 'Inserted text after "import os" in agent1/main.sh'
        """
        current_content = self._func_cat(filepath)

        if insert_after not in current_content:
            return f"Error: Pattern '{insert_after}' not found in {filepath}"

        modified_content = current_content.replace(
            insert_after, insert_after + "\n" + new_text
        )
        return self.func_write(filepath, modified_content)

    def func_append(self, filepath: str, new_text: str) -> str:
        """
        Append text to the end of a file. If the pattern is not found, an error message will be returned. After modifying, it's advisable to double-check the file for proper format.
            filepath (str): Relative path under MAIN_DIR.
            new_text (str): The text to append.

        Returns:
            str: Confirmation message.
        """
        current_content = self._func_cat(filepath)
        modified_content = current_content + "\n" + new_text
        return self.func_write(filepath, modified_content)

    def func_prepend(self, filepath: str, new_text: str) -> str:
        """
        Prepend text to the beginning of a file. If the pattern is not found, an error message will be returned. After modifying, it's advisable to double-check the file for proper format.
            filepath (str): Relative path under MAIN_DIR.
            new_text (str): The text to prepend.

        Returns:
            str: Confirmation message.
        """
        current_content = self._func_cat(filepath)
        modified_content = new_text + "\n" + current_content
        return self.func_write(filepath, modified_content)

    def func_simplify(self, message: str) -> str:
        chunks = chunk_text_by_chars(message, max_chars=10000, overlap=50)
        simplified_chunks = []
        for chunk in chunks:
            prompt = get_prompt("func_simplify_prompt.txt").format(
                message=chunk, MAIN_DIR=self.MAIN_DIR
            )
            simplified_chunk = self.llm.query(prompt)
            simplified_chunks.append(simplified_chunk)
        return "\n".join(simplified_chunks)

    def func_think(self, text: str) -> str:
        """
        Advanced reasoning helper using an auxiliary LLM for a single small task.

        This function is designed to handle **one small, self-contained problem or subtask**
        that may arise during usage. It acts as a focused reasoning module ("thinking block")
        to help the main model deepen its reasoning.

        **Important constraint:** Do not input multiple tasks or very large problems at once.
        The function must focus solely on the current small task.

        Procedure:
        1. Clear any previous tool calls.
        2. Use a separate LLM to carefully reason about this small task.
        3. Consider the task in detail based on the given `text` description.

        Important:
        - Provide as complete a description and context as possible in `text`.
        - The function may generate multiple internal reasoning steps before producing output.
        - Tool calls are managed in a temporary path, and previous calls are ignored.
        - Must not attempt to process multiple tasks or a very large task simultaneously.

        Args:
            text (str): A detailed description of the single small problem or subtask to reason about.

        Returns:
            str: The text output from the auxiliary LLM after reasoning on this small task.
        """
        tmp_tool_calls_path = os.path.join(
            self.LOCAL_TMP_PATH, "tmp_tool_calls_path.json"
        )
        tools = [
            self.func_cat,
            self.func_write,
            self.func_modify,
            self.func_insert,
            self.func_append,
            self.func_prepend,
            self.func_python,
            self.func_cmd,
            self.func_matlab,
            self.func_fetch_info_from_pmc_with_llm,
            self.func_human,
        ]
        print("[FUNC] Start func_think to clear tool call and think the problem")
        tc = Tool_Calls(
            PATH=tmp_tool_calls_path, ENV_PATH=tmp_tool_calls_path, MAX_CHAR=10000
        )
        tc.mode = "Simple"
        tc.clear()

        def stop_condition(tool_calls: List[Dict[str, Any]], *args, **kwargs):
            return len(tool_calls) == 0

        guidance = get_prompt("extra_func_think_prompt.txt")
        extra_guide_tool_call = get_func_tool_call(
            func_name="func_guide", result=guidance, guidance=guidance
        )

        result = self.llm.query_with_tools(
            prompt=get_prompt("func_think_prompt.txt").format(
                text=text, MAIN_DIR=self.MAIN_DIR
            ),
            max_steps=100,
            tools=tools,
            tc=tc,
            extra_guide_tool_call=extra_guide_tool_call,
            verbose=self.verbose,
            stop_condition=stop_condition,
        )
        print("[FUNC] Complete func_think")
        return result
    
    def func_reflect(self, text: str) -> str:
        """
        Lightweight internal reflection module.

        This function performs **pure reasoning** on a single, clearly described problem,
        without invoking any external tools or task management. It acts as a minimalistic
        "inner monologue" function that helps the model articulate its thoughts clearly.

        Usage:
            - Input a detailed problem description or question via `text`.
            - The function returns a concise, coherent reasoning output (not a direct answer).

        ⚠️ Important constraints:
            - Only one problem per call — do NOT include multiple unrelated tasks.
            - Must not attempt to execute code or call tools.
            - Designed purely for controlled, local reasoning.

        Args:
            text (str): A detailed description of the problem to think about.

        Returns:
            str: The model's reflection text, summarizing reasoning or insight.
        """
        print("[FUNC] Start func_reflect for pure reasoning")

        # 生成指令提示词
        self.llm.system_prompt=""
        reflection_prompt1 = get_prompt("reflect/func_reflect_simplify_prompt.txt").format(text=text)

        # 不调用任何工具，只进行单步思考
        result1 = self.llm.query(
            prompt=reflection_prompt1,
            verbose=self.verbose,
        )
        
        reflection_prompt2 = get_prompt("reflect/func_reflect_improve_prompt.txt").format(text=text)

        # 不调用任何工具，只进行单步思考
        result2 = self.llm.query(
            prompt=reflection_prompt2,
            verbose=self.verbose,
        )

        print("[FUNC] Complete func_reflect")
        return result1+"\n"+result2

    def add_child(self, name: str = "", description: str = "") -> str:
        """
        This function adds a new child node to the current node, but it **must not be called directly**.

        If you want to add a new child node, please use the higher-level helper function
        `help_add_child(info)` instead. `help_add_child` will handle the logic and call this
        method internally.

        Args:
            name (str): The name of the child node to be created.
            description (str, optional): A short description of the child node.
                Defaults to an empty string.

        Returns:
            str: A message indicating that this function should not be called directly.
        """
        return "Direct use of add_child is not allowed. Please use help_add_child(info) instead."

    def return_to_parent(self, result=None, success=True) -> str:
        """
        This function finalize the current node with a result and move back to its parent node., but it **must not be called directly**.

        If you want to finalize the current node, please use the higher-level helper function
        `help_return_to_parent(info)` instead. `help_return_to_parent` will handle the logic and call this
        method internally.

        Args:
            result (Any, optional): The result or message describing the outcome
                of the current node. Can include success output or failure details.
            success (bool, optional): Whether the current node is considered
                successful (True) or failed (False). Defaults to True.

        Returns:
            LogicalTree.Node: The parent node of the finalized current node.
                If the current node has no parent, it will remain as the root.
        """
        return "Direct use of return_to_parent is not allowed. Please use help_return_to_parent(info) instead."

    def func_extract_factual_info(self, text: str) -> str:
        """
        Extracts only factual information from the input text, removing all subjective,
        evaluative, or conclusive statements. The output contains objective data,
        technical details, and operational steps without any judgmental language.

        Args:
            text (str): The input text containing both factual and subjective content.

        Returns:
            str: A rewritten version of the input text containing only factual information.

        Functionality:
        1. Loads a prompt template named 'func_extract_factual_info_prompt.txt'.
        2. Inserts the input text into the prompt template.
        3. Sends the prompt to the language model (self.llm) for processing.
        4. Receives and returns the model's output, which should be a neutral,
        fact-only version of the text.
        """
        print(f"\n[FUNC] Start extracting factual information\n")
        prompt = get_prompt("func_extract_factual_info_prompt.txt").format(text=text)
        filter_text = self.llm.query(prompt=prompt, verbose=self.verbose)
        print(f"\n[FUNC] Complete extracting factual information\n")
        return filter_text

    def func_youtube_search_with_transcript(
        self,
        query: str,
        max_results: int = 5,
        order: str = "date",
        published_after: str = None,
        include_transcript: bool = True,
        max_transcript_segments: int = 20,
    ) -> str:
        """
        Search for YouTube videos and safely fetch partial transcripts.
        This version never crashes — all network, SSL, or API errors are caught and explained.
        """

        api_key = "AIzaSyCrA-ISpOpkZ_hdvrBNeEedEdxQ7HH09hM"
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "key": api_key,
            "part": "snippet",
            "type": "video",
            "q": query,
            "maxResults": max_results,
            "order": order,
        }
        if published_after:
            params["publishedAfter"] = published_after

        # ---------- 安全 requests 会话 ----------
        session = requests.Session()
        retry = Retry(connect=3, read=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        try:
            response = session.get(
                url,
                params=params,
                timeout=10,
                verify=True,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.SSLError as e:
            # SSL 问题，尝试关闭验证
            try:
                response = session.get(
                    url,
                    params=params,
                    timeout=10,
                    verify=False,  # 忽略证书验证
                )
                data = response.json()
            except Exception as e2:
                return f"⚠️ Network or SSL error (verify=False): {e2}"
        except Exception as e:
            return f"⚠️ Network or API error: {e}"

        # ---------- 解析结果 ----------
        results = []
        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            title = snippet.get("title", "[No title]")
            desc = snippet.get("description", "[No description]")
            channel = snippet.get("channelTitle", "[Unknown channel]")
            published = snippet.get("publishedAt", "[Unknown date]")
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            transcript_text = ""
            if include_transcript:
                try:
                    api = YouTubeTranscriptApi()
                    fetched = api.fetch(video_id, languages=['en', 'zh-Hans', 'zh'])
                    raw = fetched.to_raw_data()
                    partial = raw[:max_transcript_segments]
                    transcript_text = " ".join(seg.get("text", "") for seg in partial).strip()
                except Exception as e:
                    transcript_text = f"[No transcript available: {e}]"

                # 防止请求过快被封 IP
                time.sleep(random.uniform(1.0, 2.5))

            results.append(
                f"🎬 **{title}** ({channel}, {published})\n"
                f"🔗 {video_url}\n"
                f"📝 Description: {desc}\n"
                f"📜 Transcript (partial): {transcript_text[:1000]}...\n"
                "----------------------------------------\n"
            )

        if not results:
            return "No results found."

        return "\n".join(results)
    
    def _search_with_serpapi(self, query: str) -> str:
        """
        Perform search using SerpApi and return the search result. You can use this function to search, and then use web_fetch to get.

        Args:
            query (str): The search query to execute.

        Returns:
            str: The raw JSON response from SerpApi or error message.
        """
        params = {
            'q': query,
            'engine': 'google',
            'api_key': "3afc903b1ac711e49116f632133dd33ebf98f5481eed762e6cfe80e35fe7f053"
        }

        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            return response.text  # Return raw JSON as a string
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"

    def search(self, query: str) -> str:
        """
        Using SerpApi to search, please use English as query.

        Args:
            query (str): Search query, please use English as query.

        Returns:
            str: Processed results or error message
        """
        # Perform the search and get results
        raw_response = self._search_with_serpapi(query)
        
        # Try to parse the JSON response
        try:
            response_json = json.loads(raw_response)
            return json.dumps(response_json, indent=2)  # Beautified JSON output
        except json.JSONDecodeError:
            return "Error: Invalid JSON response from SerpApi."
        
    def web_fetch(self, url: str) -> str:
        """
        Fetches data from a given URL.

        Args:
            url (str): The full URL to fetch data from.

        Returns:
            str: The content of the response or an error message.
        """
        try:
            response = requests.get(url, timeout=10)  # 10秒超时
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return "Error: fetch failed (timeout > 10s)."
        except requests.exceptions.RequestException as e:
            return f"Error: fetch failed ({e})."
        except ValueError:
            return "Error: Invalid JSON response."

def chunk_text_by_chars(text: str, max_chars: int = 500, overlap: int = 50):
    """
    Split text into chunks based on characters, with optional overlap.

    Args:
        text (str): The text to split.
        max_chars (int): Maximum characters per chunk.
        overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    start = 0
    end = 0
    text_len = len(text)

    while end < text_len:
        end = min(start + max_chars, text_len)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0

    if len(chunks) == 0:
        chunks = ["[No output generated]"]

    return chunks


def clean_training_logs(log_text: str) -> str:
    cleaned_lines = []
    seen_progress = None

    for line in log_text.splitlines():
        # 跳过带有 \b 或 \r 的动态刷新行
        if "\b" in line or "\r" in line:
            continue

        # 匹配 keras/torch 风格的进度条
        if re.match(r"^\d+/\d+ \[.*\] - ETA: .* - loss: .*", line):
            seen_progress = line  # 只保留最后一次
            continue

        cleaned_lines.append(line)

    # 如果有进度信息，补上最后一次
    if seen_progress:
        cleaned_lines.append(seen_progress)

    return "\n".join(cleaned_lines)
