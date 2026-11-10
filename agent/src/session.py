"""
会话持久化：一个会话一个json文件，每次有消息输出则重写整个文件
LLM 一次通常只调一个工具（除非多个工具彼此独立无依赖）
{
  "metadata": {
    "id": "b4ac4552",                              // 会话 ID（8 位随机 hex）
    "model": "deepseek-chat",                       // 模型
    "cwd": "D:\\agent\\my-project",                 // 启动时的目录
    "startTime": "2026-08-03T01:36:00Z"             // 会话开始时间
  },
  "messages": [
    // system 消息
       对应SystemMessage(".......")
    {"t": "s", "c": "你是 Claude Code Agent\n\n# Environment\n..."},

    // 用户: "帮我读 a.py"
       对应HumanMessage("......")
    {"t": "h", "c": "<system-reminder>...</system-reminder>\n\n帮我读 a.py"},

    // 模型回复: "好的" + 只调read_file 工具
      对应AIMessage(content="",tool_calls=[{"name":xxx,"args"={xxx},"id"=xx}])
    {"t": "a", "c": "好的，我来读一下", "tc": [
        {"name": "read_file", "args": {"file_path": "a.py"}, "id": "tool_1", "type": "tool_call"}
    ]},

    // 工具结果
        对应ToolMessage(content="...",tool_call_id="...")
    {"t": "tl", "c": "1 | print('hello')\n2 | print('world')", "id": "tool_1"},

    // 模型继续: "a.py 内容是..."
       对应AIMessage(content=".......")
    {"t": "a", "c": "a.py 文件内容如下：\n```python\nprint('hello')\nprint('world')\n```"},

    // 用户第二轮: "改个 bug"
    {"t": "h", "c": "改个 bug"},

    // 模型调了 edit_file
    {"t": "a", "c": "", "tc": [
        {"name": "edit_file", "args": {"file_path": "a.py", "old_string": "print('world')", "new_string": "print('hello world')"}, "id": "tool_2", "type": "tool_call"}
    ]},

    // 工具结果
    {"t": "tl", "c": "Successfully edited a.py\n\n@@ -2,1 +2,1 @@\n- print('world')\n+ print('hello world')", "id": "tool_2"},

    // 模型确认
    {"t": "a", "c": "已修复，world 改成 hello world 了"}
  ]
}

"""



from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 会话文件存在用户项目目录下，不跟 Agent 源码混在一起
def _project_root() -> Path:
    return Path.cwd()

SESSION_DIR = _project_root() / ".claude-code-agent" / "sessions"

#=========--resume时，先获取最新会话的id；接着传入sessionid，获取对应文件内容=================
def get_latest_session_id() -> str | None:
    sessions = _list_sessions()
    if not sessions:
        return None
    sessions.sort(key=lambda s: s.get("startTime", ""), reverse=True)
    return sessions[0].get("id")

#扫描整个sessions目录，只取metedata加入数组
def _list_sessions() -> list[dict[str, Any]]:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "metadata" in data:
                results.append(data["metadata"])
        except Exception:
            pass
    return results

#传入id，恢复内容
def load_session(session_id: str) -> dict[str, Any] | None:
    path = SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


#================会话结束后，保存================================
def save_session(session_id: str, data: dict[str, Any]) -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    (SESSION_DIR / f"{session_id}.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")