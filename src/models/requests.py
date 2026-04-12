# src/models/requests.py
from pydantic import BaseModel
from typing import Optional, Any, Literal
from src.config import PLAYBOOK_ROOT


# === 数据模型 ===
class PlaybookSuggestNextStepReq(BaseModel):
    playbook_root: str = PLAYBOOK_ROOT
    model_name: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"

    assumptions: list[str] = []
    current_condition: str

# DOC-BEGIN id=manual/relay/remote-plan-models#1 type=design v=1
# summary: 远端中转器的“计划数据 set/get”请求模型
# intent: 该中转器只负责存取，不对 plan 的具体结构做强约束（避免前端迭代被后端 schema 卡死）；
#   仅增加可选 source 方便排查“是谁写入的”（例如主前端/测试脚本/另一个前端）。
class RemotePlanSetReq(BaseModel):
    plan: Any
    source: Optional[str] = None

class RemotePlanGetReq(BaseModel):
    pass
# DOC-END id=manual/relay/remote-plan-models#1

class ArticleDeleteBlogReq(BaseModel):
    article_id: str
    
class TaskGetTerminalStatusReq(BaseModel):
    task_id: str
    
# DOC-BEGIN id=models/requests/article-extract-images-req#1 type=design v=2
# summary: ArticleExtractImagesReq 包含文章ID和可选的Adobe PDF Services凭据；
#   PDF图片提取需要client_id和client_secret，HTML提取不需要；
#   可选字段允许前端按需传递凭据，同时保持向后兼容
# intent: 支持PDF和HTML两种格式的图片提取；PDF需要Adobe API凭据（每月500次免费），
#   HTML不需要额外凭据；保持模型简洁，避免强制传递不需要的参数
class ArticleExtractImagesReq(BaseModel):
    article_id: str
    pdf_services_client_id: Optional[str] = None
    pdf_services_client_secret: Optional[str] = None
# DOC-END id=models/requests/article-extract-images-req#1

class ArticleUpsertHTML(BaseModel):
    task_id: Optional[str] = None
    title: str
    html: str
    article_id: Optional[str] = None # 如果传了文件名（如 "test.html"），就用这个

class ArticleListReq(BaseModel):
    status: Optional[Literal["processing","completed","all"]] = "all"

class ArticleGetHtmlReq(BaseModel):
    article_id: str

class TaskSelectArticleReq(BaseModel):
    task_id: str
    article_id: str
    
class PinnedCode(BaseModel):
    hash: str
    filename: str
    content: str

class AvailableLLM(BaseModel):
    id: str          # 唯一标识，如 "fast" / "powerful"
    label: str       # 显示名，如 "GPT-4o-mini"
    model_name: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"

class LLMRequestData(BaseModel):
    model_name: str
    question: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"
    task_id: Optional[str] = "1"
    system_prompt_mode: Optional[str] = "default"
    enable_fc: Optional[bool] = False
    trim_history: Optional[bool] = False
    pinned_codes: Optional[list[PinnedCode]] = []
    is_dev_mode: Optional[bool] = False
    available_llms: Optional[list[AvailableLLM]] = []
    reasoning_enabled: Optional[bool] = False

class ActionRequest(BaseModel):
    action: str
    data: dict

class StopData(BaseModel):
    task_id: str
    
# DOC-BEGIN id=models/requests/article-generate-blog-req#2 type=design v=2
# summary: ArticleGenerateBlogReq 包含博客生成所需的全部参数：文章定位(article_id/task_id)、
#   主LLM配置(model_name/api_key/llm_url)、风格(style)、可选的Adobe PDF Services凭据、
#   以及可选的后处理LLM配置(refine_model_name/refine_api_key/refine_llm_url)。
#   后处理LLM是单模态，用于精修和概念补充，不处理图片。
# intent: 前端在生成blog时统一传递所有参数，后端按需使用；凭据可选保持向后兼容；
#   后处理参数可选，不传则跳过后处理步骤
class ArticleGenerateBlogReq(BaseModel):
    task_id: str
    article_id: str
    model_name: str
    api_key: str
    llm_url: Optional[str] = None
    style: Optional[Literal["math", "normal", "rigorous"]] = "math"
    pdf_services_client_id: Optional[str] = None
    pdf_services_client_secret: Optional[str] = None
    # 后处理LLM配置（单模态，用于精修和概念补充）
    refine_model_name: Optional[str] = None
    refine_api_key: Optional[str] = None
    refine_llm_url: Optional[str] = None
# DOC-END id=models/requests/article-generate-blog-req#2


# DOC-BEGIN id=models/requests/tts-req#1 type=design v=1
# summary: ArticleGenerateTTSReq 包含文章定位(article_id)、LLM配置(model_name/api_key/llm_url)
#   和Fish Audio配置(fish_api_key/reference_id)；前端必须传入fish_api_key和reference_id
# intent: TTS生成需要两阶段调用：先调LLM改写文本，再调Fish Audio生成音频；
#   两个服务使用不同的API key，因此分开传递；reference_id决定说话人声音，
#   不同用户/场景可能选择不同声音
class ArticleGenerateTTSReq(BaseModel):
    article_id: str
    model_name: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"
    fish_api_key: str
    reference_id: str
# DOC-END id=models/requests/tts-req#1

class ArticleDeleteTTSReq(BaseModel):
    article_id: str


class NewsManualFetchReq(BaseModel):
    pass

class NewsAutoToggleReq(BaseModel):
    enable: bool

class NewsListReq(BaseModel):
    page: int = 1
    page_size: int = 20
    category: str | None = None

