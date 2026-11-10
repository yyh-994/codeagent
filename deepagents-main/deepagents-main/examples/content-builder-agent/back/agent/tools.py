"""所有 Agent 工具 —— 搜索、生图、RAG、文件"""

import os
from pathlib import Path

from back.common.config import IMAGE_GEN_MODEL, SEARCH_MAX_RESULTS
from typing import Literal
from langchain_core.tools import tool
from back.common.exceptions import ExternalServiceException
from langchain_core.runnables import RunnableConfig

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ── 搜索 ──

@tool
def web_search(query: str, max_results: int = SEARCH_MAX_RESULTS,
               tbm: Literal[None, "nws"] = None) -> dict:
    """通过 SerpApi 搜索互联网获取最新信息。"""
    try:
        from serpapi import GoogleSearch
        api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPAPI_KEY")
        if not api_key: return {"success": False, "error": "SERPAPI key not set"}
        params = {"q": query, "num": max_results, "api_key": api_key, "engine": "google"}
        if tbm:
            params["tbm"] = tbm
        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])
        # answer_box: Google 精选摘要，简单事实查询时 LLM 可直接用
        # 类型: "snippet"(段落) / "answer"(短回答) / "list"(列表) / "table"(表格)
        answer = results.get("answer_box", {})
        return {
            "success": True,
            "results": [{"title": r.get("title",""), "url": r.get("link",""), "content": r.get("snippet","")} for r in organic],
            "answer_box": {"title": answer.get("title",""), "content": answer.get("answer") or answer.get("snippet",""), "type": answer.get("type","")} if answer else None,
        }
    except Exception as e:
        raise ExternalServiceException("SerpAPI", str(e))


# ── 生图 ──

@tool
def generate_image(
    prompt: str,
    topic: str,
) -> dict:
    """生成文章配图。prompt: 英文画面描述（按 skill 中的生图规范撰写），topic: 写作主题，图片存为 back/output/blog_output/<topic>.png。"""
    try:
        from huggingface_hub import InferenceClient
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            return {"success": False, "error": "HUGGINGFACE_TOKEN not set"}
        client = InferenceClient(token=token)
        image = client.text_to_image(prompt, model=IMAGE_GEN_MODEL)
        path = os.path.join(ROOT_DIR, "back", "output", "blog_output", f"{topic}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
        return {"success": True, "path": path, "message": f"Image saved to {path}"}
    except Exception as e:
        raise ExternalServiceException("HuggingFace", str(e))


# ── 个人课表查询 ──

@tool
async def query_my_schedule(
    time_scope: Literal[
        "today", "tomorrow", "this_week", "next_week", "date", "week", "term",
    ] | None = None,
    query_date: str | None = None,
    week_no: int | None = None,
    weekday: int | None = None,
    course_name: str | None = None,
) -> dict:
    """查询我自己的课表，后端根据北京时间和唯一学期校历计算日期与教学周。

    time_scope: today 今天、tomorrow 明天、this_week 本周、next_week 下周、
      date 指定日期、week 指定教学周、term 整学期。
    query_date: 指定日期 YYYY-MM-DD。week_no: 第几教学周。weekday: 周一=1，周日=7。
    course_name: 可选课程名筛选。未给时间默认今天；仅指定星期默认本周。
    不需要姓名、学号或学期。所有参数都可省略，不要编造缺失条件。
    返回课程、日期、教学周、星期、节次、实际起止时间和教室。
    日期超出学期会返回范围说明，不代表当天一定没课。
    """
    from back.service.school.timetable_service import query_schedule
    return await query_schedule(
        time_scope=time_scope, query_date=query_date, week_no=week_no,
        weekday=weekday, course_name=course_name,
    )


# ── 旧本地查询接口（保留兼容，查询 Agent 通过 MCP 发现工具）──

@tool
async def query_school_db(
    sql: str, *, params: dict[str, str | int | float | bool | None] | None = None,
) -> list[list]:
    """旧本地调用兼容：执行三张课表业务表的 SELECT，值通过 :name 与 params 绑定。

    查询 Agent 实际使用的是 FastMCP 服务中同名的工具，不绑定本地版本。
    """
    from back.service.school.school_service import execute_sql
    return await execute_sql(sql, params=params)


# ── RAG ──

@tool
def upload_and_index_pdf(pdf_path: str, config: RunnableConfig):
    """上传并索引一个 PDF 到本地知识库。pdf_path: PDF 文件路径。"""
    from back.service.rag.knowledge_service import knowledge_build
    uid = config.get("configurable", {}).get("user_id", "")
    return knowledge_build(pdf_path, uid)


@tool
def search_knowledge_base(query: str, config: RunnableConfig):
    """从本地知识库（已上传索引的 PDF）检索相关内容（仅检索你自己的知识库）。即使结果中没有原词出现，也应当基于语义相关的片段给出答案，不可直接说"未找到"。"""
    from back.service.rag.search_service import search
    uid = config.get("configurable", {}).get("user_id", "")
    return search(query, uid)


# ── 子 Agent 配置 ──

import yaml

def load_subagents(config_path: Path) -> list: #yaml->list[dict1,dict2...]
    """从 YAML 加载子 Agent 定义。tools 中声明的是从主 Agent 显式注入的工具，
    文件操作工具（write_file/read_file/edit_file/ls/glob/grep）由 Middleware 自动注入，无需声明。"""
    #yaml里的工具名仅是字符串，这里做了把字符串转工具
    tool_map = {"web_search": web_search}
    with open(config_path) as f:
        subagents_file = yaml.safe_load(f)
    subagents = []
    for key, value in subagents_file.items():
        sa = {"name": key, **value}#sa = { "name": "reviewer", "description": "...", "system_prompt": "..." }
        if "tools" in sa:
            sa["tools"] = [tool_map[t] for t in sa["tools"]]
        subagents.append(sa)
    return subagents
