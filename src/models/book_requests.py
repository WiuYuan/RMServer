# src/models/book_requests.py
# DOC-BEGIN id=book/requests#1 type=model v=1
# summary: 书籍阅读模块的请求模型定义，包括上传、列表、获取、删除、获取页面、生成注释等操作的参数结构
# intent: 统一使用Pydantic模型校验输入参数，与现有article_handlers.py中ArticleListReq等保持一致的风格
from pydantic import BaseModel
from typing import Optional, List


class BookListReq(BaseModel):
    """扫描books目录，返回所有已发现的PDF书籍"""
    task_id: Optional[str] = None


class BookGetReq(BaseModel):
    book_id: str


class BookDeleteReq(BaseModel):
    book_id: str


class BookGetPagesReq(BaseModel):
    book_id: str
    pages: List[int]  # 1-based 页码列表


class BookGenerateAnnotationReq(BaseModel):
    book_id: str
    pages: List[int]  # 要注释的页码范围
    model_name: str
    api_key: str
    llm_url: Optional[str] = None


class BookGetAnnotationReq(BaseModel):
    book_id: str
# DOC-END id=book/requests#1