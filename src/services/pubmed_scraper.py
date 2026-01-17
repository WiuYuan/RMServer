import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from typing import List, Tuple
import json
from src.services.llm import LLM
import tiktoken
import os
import ast


def extract_article_ids(html):
    soup = BeautifulSoup(html, "html.parser")
    article_ids = set()

    # 找所有 <a> 标签
    for a in soup.find_all("a"):
        for attr in a.attrs.values():
            if isinstance(attr, str):
                match = re.search(r"article_id=(\d+)", attr)
                if match:
                    article_ids.add(match.group(1))

    return list(article_ids)


def extract_abstract(html):
    soup = BeautifulSoup(html, "html.parser")
    abstract_div = soup.find("div", class_="abstract-content selected")

    if not abstract_div:
        return "Abstract not found."

    # 提取所有 <p> 标签内容并合并为一个段落
    paragraphs = abstract_div.find_all("p")
    abstract_text = "\n".join(p.get_text(strip=True) for p in paragraphs)

    return abstract_text


def extract_total_pages(html):
    # 解析 HTML
    soup = BeautifulSoup(html, "html.parser")

    # 查找目标标签
    label = soup.find("label", class_="of-total-pages")
    if not label:
        raise ValueError("未找到 <label class='of-total-pages'> 标签")

    # 提取文本并用正则匹配数字
    text = label.get_text(strip=True)  # 例如 "of 114"
    match = re.search(r"\b(\d+)\b", text)  # 匹配连续数字

    if not match:
        raise ValueError("未找到数字")

    return int(match.group())


def get_all_article_ids(query, max_pages=1e6, delay=1.0):
    """
    生产者函数：逐步生成 PubMed 文章 ID（生成器版本）

    参数:
        query (str): 搜索关键词
        max_pages (int): 最大获取页数
        delay (float): 每页之间延迟秒数（默认1.0）

    生成:
        str: 每次 yield 一个文章 ID
    """
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"

    # 请求第一页获取总页数
    params = {"term": query, "page": 1}
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    html = response.text

    # total_pages = extract_total_pages(html)
    # total_pages = min(total_pages, max_pages)
    total_pages = max_pages
    print(f"Total pages to fetch: {total_pages}")

    for page in range(1, total_pages + 1):
        params["page"] = page
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        html = response.text
        ids = extract_article_ids(html)

        for article_id in ids:
            yield article_id  # 逐个生成文章ID

        time.sleep(delay)  # 遵守爬虫礼貌性延迟


def is_review_article(html: str) -> bool:
    """
    判断 HTML 是否包含 'Review' 类型文章。

    参数:
        html (str): HTML 文本

    返回:
        bool: 是否为 Review 类型
    """
    soup = BeautifulSoup(html, "html.parser")
    pub_type_divs = soup.find_all("div", class_="publication-type")
    for div in pub_type_divs:
        if div.get_text(strip=True).lower() == "review":
            return "Yes"
    return "No"


def remove_think(text):
    """
    从文本中移除 <think>...</think> 部分，并清理开头和结尾的空白字符，
    最终返回剩余内容

    参数:
    text (str): 输入的文本字符串

    返回:
    str: 处理后的干净文本（如"Yes"）
    """
    # 移除think部分
    start_tag = "<think>"
    end_tag = "</think>"

    start_idx = text.find(start_tag)
    if start_idx != -1:
        end_idx = text.find(end_tag, start_idx)
        if end_idx != -1:
            # 移除think部分（包括标签）及其前后的空白
            text = text[:start_idx] + text[end_idx + len(end_tag) :]

    # 清理开头和结尾的空白字符
    return text.strip()


def find_pmc_free_article(html_content):
    """
    Find PMC free article link in HTML content.

    Args:
        html_content (str): HTML content to search

    Returns:
        str: PMC free article URL if found, otherwise "No"
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all links with class containing 'pmc' and 'link-item'
    pmc_links = soup.find_all(
        "a", class_=lambda x: x and "pmc" in x.split() and "link-item" in x.split()
    )

    for link in pmc_links:
        if "Free full text at PubMed Central" in link.get("title", ""):
            return link["href"]

    return "No"


def chunk_text_by_tokens(text, model_name="gpt-4", max_tokens=3000):
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]


def generate_prompt_from_chunk(chunk):
    prompt = f"""Please answer the following questions strictly based on the content of the article provided below. Only rely on explicitly stated information. If the article does not mention something, respond with "Not mentioned".

Article content:
{chunk}

Please answer the following questions:
1. Is this article an original research work that trains a model on data (as opposed to a review or theoretical discussion)?
   - If the article describes training a model on specific datasets, answer "Yes".
   - If it is a review, commentary, or theoretical framework without model training, answer "No".
   - If unclear, answer "Not mentioned".

2. Does the article provide publicly available trained models (e.g., checkpoints/weights)?
   - If it mentions model availability on platforms like GitHub, HuggingFace, or institutional websites, specify the details.
   - Otherwise, respond with "Not mentioned".

3. Does the article release implementation code?
   - If it refers to a code repository (e.g., GitHub/GitLab link) or includes code in appendices, provide specific details.
   - Otherwise, respond with "Not mentioned".

4. Is the dataset used publicly available?
   - If the article clearly states the dataset name and how to access it (e.g., URL or database ID), provide detailed information.
   - If the dataset is private, respond with "Private dataset".
   - Otherwise, respond with "Not mentioned".

5. Key characteristics of the dataset (if publicly available):
   a. Type of image data used (e.g., MRI, PET)
      - Specify the imaging modality if mentioned.
   
   b. Type of genomic data used (e.g., EGFR mutation, MGMT methylation)
      - Indicate the specific genetic/molecular markers, if available.
   
   c. Sample size
      - Report the total number of samples, and if available:
        - The number of samples in each group (e.g., training/validation/test sets)
        - Any stratification by disease status or mutation subtype
        - Whether the study includes multi-modal data (e.g., paired image + genomic data), and the sample size for each
        - The collection time span if reported (e.g., samples collected from 2015–2020)
   
   d. Target disease/problem (e.g., NSCLC)
      - Mention the disease or clinical condition the dataset is focused on.
   
   e. Data collection method (e.g., public database, in-house collection)
      - Specify the source of the data: a public dataset (include name if given), a hospital/institution, or self-collected.
   
   f. Annotation method (e.g., expert-labeled, automatically generated)
      - Describe how the labels or outcomes were assigned to the data, including whether annotation was done by human experts, via automated pipelines, or using other criteria.

Please respond in JSON format as follows:
{{
    "is_original_model_training_study": "answer to question 1"
    "public_model": "answer to question 2",
    "public_code": "answer to question 3",
    "public_dataset": "answer to question 4",
    "dataset_details": {{
        "image_data_type": "answer to question 5.a",
        "genome_data_type": "answer to question 5.b",
        "sample_size": "answer to question 5.c",
        "disease": "answer to question 5.d",
        "collection_method": "answer to question 5.e",
        "annotation": "answer to question 5.f"
    }}
}}"""

    return prompt


def generate_article_prompt(article_text, model_name="gpt-4", max_tokens=3000):
    cleaned_text = remove_references(article_text)
    chunks = chunk_text_by_tokens(
        cleaned_text, model_name=model_name, max_tokens=max_tokens
    )
    prompts = [generate_prompt_from_chunk(chunk) for chunk in chunks]
    return prompts


def extract_json(response: str):
    json_str = remove_think(response)
    start = json_str.find("{")
    end = json_str.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_candidate = json_str[start : end + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败:")
            print(f"提取片段: {json_candidate}")
            print(f"错误: {e}")
    return {}


def generate_merge_prompt(json_outputs: List[str]) -> str:
    return f"""You are an assistant that merges structured JSON answers.

Below are several JSON outputs generated by analyzing different parts ("chunks") of a long article. Each output contains partial answers to the same set of questions, based on the content of one chunk.

Your task is to **merge them into a single complete answer**, by following these rules:

### Rules for merging:
1. If at least one JSON contains a valid answer (i.e., anything other than "Not mentioned"), use that value.
2. If multiple answers exist and they do not conflict, you may keep the more complete one.
3. If conflicting information appears (e.g., different sample sizes), try to merge both if possible or mention both (e.g., "300–500 samples").
4. If all values are "Not mentioned", then keep "Not mentioned".

### Expected Output:
Return a **single valid JSON** object in the following format (do not explain anything):

{{
    "is_original_model_training_study": "answer to question 1"
    "public_model": "answer to question 2",
    "public_code": "answer to question 3",
    "public_dataset": "answer to question 4",
    "dataset_details": {{
        "image_data_type": "answer to question 5.a",
        "genome_data_type": "answer to question 5.b",
        "sample_size": "answer to question 5.c",
        "disease": "answer to question 5.d",
        "collection_method": "answer to question 5.e",
        "annotation": "answer to question 5.f"
    }}
}}

### JSON Outputs to merge:
{json.dumps(json_outputs, indent=2)}
"""


def merge_llm_outputs(llm: LLM, json_list: List[str]):
    """
    从多个 chunk 的 LLM 输出中整合回答。
    只要有一个 chunk 提供了有效答案，就保留。
    """
    merged = {
        "public_model": "Not mentioned",
        "public_code": "Not mentioned",
        "public_dataset": "Not mentioned",
        "dataset_details": {
            "image_data_type": "Not mentioned",
            "genome_data_type": "Not mentioned",
            "sample_size": "Not mentioned",
            "disease": "Not mentioned",
            "collection_method": "Not mentioned",
            "annotation": "Not mentioned",
        },
        "is_original_model_training_study": "Not mentioned",
    }

    for output in json_list:
        for key in [
            "public_model",
            "public_code",
            "public_dataset",
            "is_original_model_training_study",
        ]:
            if output.get(key, "Not mentioned") != "Not mentioned":
                merged[key] = output[key]

        for subkey in merged["dataset_details"]:
            if (
                output.get("dataset_details", {}).get(subkey, "Not mentioned")
                != "Not mentioned"
            ):
                merged["dataset_details"][subkey] = output["dataset_details"][subkey]

    return merged


def fetch_info_for_article(llm: LLM, article, verbose=False):
    prompts = generate_article_prompt(article)
    json_list = []
    for _, prompt in enumerate(prompts):
        json_list.append(remove_think(llm.query(prompt, verbose)))
    prompt = generate_merge_prompt(json_list)
    return extract_json(llm.query(prompt, verbose))


def fetch_pmc_article(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查HTTP错误
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def fetch_info_for_one_pmc_article(llm: LLM, pmc_url, verbose=True):
    article = remove_references(extract_main_article_body(fetch_pmc_article(pmc_url)))
    return fetch_info_for_article(llm, article, verbose=verbose)


def fetch_info_for_pmc(llm: LLM, query, filepath, max_pages=100000, delay=1.0):
    existing_info = {}
    if os.path.exists(filepath):
        existing_info = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = ast.literal_eval(line.strip())  # 安全解析Python字面量
                    existing_info[data["pmid"]] = data
                except Exception as e:
                    print("跳过解析错误:", e)
                    continue
    with open(filepath, "a", encoding="utf-8") as f:
        for pmid in get_all_article_ids(query, max_pages, delay):
            if pmid in existing_info:
                # 如果已有结果，直接用
                info = existing_info[pmid]
                print(info)
                continue
            info = {
                "pmid": pmid,
                "is_review": None,
                "is_radiogenomics": None,
                "is_pmc_free": None,
                "pmc_url": None,
            }
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            response = requests.get(url, timeout=10)
            html = response.text
            info["is_review"] = is_review_article(html)
            if info["is_review"] == "No":
                abstract = extract_abstract(html)
                prompt = (
                    'Please strictly answer only "Yes" or "No" without any explanations or additional content.\n'
                    "Now determine whether the following abstract belongs to radiogenomics research (must actually contain genomic data):\n\n"
                    f"Abstract:\n{abstract}\n\n"
                    "Judgment:"
                )
                info["is_radiogenomics"] = remove_think(
                    llm.query(prompt, verbose=False)
                )
                if info["is_radiogenomics"] == "Yes":
                    pmc_url = find_pmc_free_article(html)
                    info["is_pmc_free"] = "No" if pmc_url == "No" else "Yes"
                    if info["is_pmc_free"] == "Yes":
                        info["pmc_url"] = pmc_url
                        # response = requests.get(pmc_url, timeout=10)
                        # html = response.text
                        article = remove_references(
                            extract_main_article_body(fetch_pmc_article(pmc_url))
                        )
                        info.update(fetch_info_for_article(llm, article))
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
            print(info)


def extract_main_article_body(html):
    """
    从HTML文件中提取 <section class="body main-article-body"> 下的所有文本内容

    参数:
        html_file_path (str): HTML文件的路径

    返回:
        str: 该section下所有文本的合并结果
    """
    soup = BeautifulSoup(html, "html.parser")
    main_section = soup.find("section", class_="body main-article-body")
    all_text = " ".join(main_section.stripped_strings)
    return all_text


def remove_references(text):
    """
    Remove everything starting from the first occurrence of 'References' or similar variants.
    """
    # 强匹配，忽略大小写，从 'References' 起直到文末都删掉
    pattern = r"(References\b.*?$)([\s\S]*)"  # References开头，后面任何内容都删
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return text[: match.start(1)].strip()
    return text
