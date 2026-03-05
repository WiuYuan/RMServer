from pydantic import BaseModel
from typing import Optional, Any, Literal


# === 数据模型 ===
class PlaybookSuggestNextStepReq(BaseModel):
    playbook_root: str = "/home/ubuntu/workspace/data/playbook"
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
    
class ArticleExtractImagesReq(BaseModel):
    article_id: str

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

class LLMRequestData(BaseModel):
    model_name: str
    question: str
    api_key: str
    llm_url: Optional[str] = "https://api.deepseek.com/v1/chat/completions"
    task_id: Optional[str] = "1"
    system_prompt_mode: Optional[str] = "default"
    enable_fc: Optional[bool] = False
    pinned_codes: Optional[list[PinnedCode]] = []

class ActionRequest(BaseModel):
    action: str
    data: dict

class StopData(BaseModel):
    task_id: str
    
class ArticleGenerateBlogReq(BaseModel):
    task_id: str
    article_id: str
    model_name: str
    api_key: str
    llm_url: Optional[str] = None
    style: Optional[Literal["math", "normal", "rigorous"]] = "math"

