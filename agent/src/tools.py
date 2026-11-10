"""Tool definitions and execution — integrated with LangGraph.

Tool definitions follow the LangChain tool pattern. Execution handlers are
async and receive a mutable state channel for read-before-edit tracking.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from .skills import build_skill_descriptions, execute_skill

# ─── Permission modes ──────────────────────────────────────

PermissionMode = str

EDIT_TOOLS = {"write_file", "edit_file"}
CONCURRENCY_SAFE_TOOLS = {"read_file", "list_files", "grep_search"}
# plan 模式下允许调用的只读 MCP 工具白名单（完整前缀名，如联网搜索）
PLAN_ALLOWED_MCP_TOOLS = frozenset({
    "mcp__serpapi__web_search",
})

IS_WIN = sys.platform == "win32"

# 在graph.py _call_llm：转成 OpenAI function calling 格式绑给 LLM，告诉它"你能调这些函数"
tool_definitions = [
    {
        "name": "memory_search",
        "description": "Search this project's long-term memories using vector similarity. Returns IDs, types, content and scores. Similarity alone does not establish a duplicate or contradiction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "threshold": {"type": "number", "minimum": -1, "maximum": 1},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_save",
        "description": "Save one independent long-term fact after checking it with memory_search. Skip unchanged facts. Omit id to create, supply an existing search result id to update. Finish this fact before processing the next. Content and vector are stored together in Milvus. A timeout may have an uncertain outcome: search before retrying.",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": ["user", "feedback", "project", "reference"]},
                "content": {"type": "string", "minLength": 1, "maxLength": 4000},
            },
            "required": ["type", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "memory_delete",
        "description": "Delete a memory and its vector by an existing ID when the user requests forgetting it.",
        "input_schema": {
            "type": "object",
            "properties": {"memory_id": {"type": "string"}},
            "required": ["memory_id"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to read"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to write"},
                "content": {"type": "string", "description": "The content to write to the file"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing an exact string match with new content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path to the file to edit"},
                "old_string": {"type": "string", "description": "The exact string to find and replace"},
                "new_string": {"type": "string", "description": "The string to replace it with"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern. Returns matching file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": 'Glob pattern to match files (e.g., "**/*.ts")'},
                "path": {"type": "string", "description": "Base directory to search from."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_search",
        "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file to search in."},
                "include": {"type": "string", "description": 'File glob pattern to include (e.g., "*.ts")'},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_shell",
        "description": "Execute a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "number", "description": "Timeout in milliseconds (default: 30000)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "skill",
        "description": build_skill_descriptions(),
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "The skill name from the available skills listed in the tool description"},
                "args": {"type": "string", "description": "Optional arguments to pass to the skill template"},
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "agent",
        "description": "Launch a read-only sub-agent to search and explore the codebase. The sub-agent has isolated context and returns its findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Detailed task instructions for the sub-agent"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "enter_plan_mode",
        "description": "Enter plan mode: read-only exploration, write a plan file before making changes.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "exit_plan_mode",
        "description": "Exit plan mode: submit your completed plan for user approval before execution.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ----------------8个工具声明，关于plan的2个工具执行内容在graph.py---------------------------
def _read_file(inp: dict) -> str:
    try:
        content = Path(inp["file_path"]).read_text(encoding="utf-8", errors="replace") #str
        lines = content.split("\n") #list
        return "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines)) #str
    except Exception as e:
        return f"Error reading file: {e}"

def _execute_skill(inp: dict) -> str:
    """处理 skill 工具调用：返回渲染后的流程模板，主 agent 照着步骤继续执行。"""
    result = execute_skill(inp.get("skill_name", ""), inp.get("args", ""))
    if not result:
        return f"Unknown skill: {inp.get('skill_name', '')}"
    return f'[Skill "{inp.get("skill_name", "")}" activated]\n\n{result["prompt"]}'

def _lint_check(file_path: str, content: str) -> str | None:
    """写前静态检查（Gate 1）：对 Python 文件做语法检查。

    用内置 compile() 对源码编译（只检查语法，不写盘、不执行）。
    非 .py 文件跳过（返回 None，即不拦截）。

    Returns:
        None 表示通过；否则返回错误描述字符串。
    """
    if not file_path.endswith(".py"):
        return None
    try:
        compile(content, file_path, "exec")
        return None
    except SyntaxError as e:
        return f"语法错误 (line {e.lineno}): {e.msg}"
    except Exception as e:
        return f"静态检查失败: {e}"


def _write_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        content = inp["content"]
        # Gate 1 写前静态检查：语法错误则拒绝写入，错误回传 Agent 修改
        lint_err = _lint_check(inp["file_path"], content)
        if lint_err:
            return f"Write rejected by lint gate:\n{lint_err}\n\n请修复语法错误后重试。"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        lines = inp["content"].split("\n")
        line_count = len(lines)
        preview = "\n".join(f"{i+1:4d} | {l}" for i, l in enumerate(lines[:30]))
        trunc = f"\n  ... ({line_count} lines total)" if line_count > 30 else ""
        return f"Successfully wrote to {inp['file_path']} ({line_count} lines)\n\n{preview}{trunc}"
    except Exception as e:
        return f"Error writing file: {e}"

def _edit_file(inp: dict) -> str:
    try:
        path = Path(inp["file_path"])
        content = path.read_text(encoding="utf-8", errors="replace")
        actual = _find_actual_string(content, inp["old_string"])
        if not actual:
            return f"Error: old_string not found in {inp['file_path']}"
        count = content.count(actual)
        if count > 1:
            return f"Error: old_string found {count} times in {inp['file_path']}. Must be unique."
        new_content = content.replace(actual, inp["new_string"], 1)
        # Gate 1 写前静态检查：对修改后的完整内容做语法检查
        lint_err = _lint_check(inp["file_path"], new_content)
        if lint_err:
            return f"Edit rejected by lint gate:\n{lint_err}\n\n请修复语法错误后重试。"
        path.write_text(new_content, encoding="utf-8")
        diff = _generate_diff(content, actual, inp["new_string"])
        quote_note = " (matched via quote normalization)" if actual != inp["old_string"] else ""
        return f"Successfully edited {inp['file_path']}{quote_note}\n\n{diff}"
    except Exception as e:
        return f"Error editing file: {e}"

def _list_files(inp: dict) -> str:
    try:
        base = Path(inp.get("path") or ".")
        pattern = inp["pattern"]
        files = []
        extra = 0
        for p in base.glob(pattern):
            if p.is_file():
                rel = str(p.relative_to(base) if base != Path(".") else p)
                if any(part == "node_modules" or part.startswith(".") for part in Path(rel).parts):
                    continue
                if len(files) < 200:
                    files.append(rel)
                else:
                    extra += 1
        if not files:
            return "No files found matching the pattern."
        result = "\n".join(files)
        if extra:
            result += f"\n... and {extra} more"
        return result
    except Exception as e:
        return f"Error listing files: {e}"

def _grep_search(inp: dict) -> str:
    pattern = inp["pattern"]
    path = inp.get("path") or "."
    include = inp.get("include")
    if not IS_WIN:
        try:
            args = ["grep", "--line-number", "--color=never", "-r"]
            if include:
                args.append(f"--include={include}")
            args.extend(["--", pattern, path])
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            if result.returncode == 1:
                return "No matches found."
            if result.returncode == 0:
                lines = [l for l in result.stdout.split("\n") if l]
                output = "\n".join(lines[:100])
                if len(lines) > 100:
                    output += f"\n... and {len(lines) - 100} more matches"
                return output
        except Exception:
            pass
    return _grep_python(pattern, path, include)

def _run_shell(inp: dict) -> str:
    try:
        timeout_ms = inp.get("timeout", 30000)
        timeout_s = timeout_ms / 1000
        result = subprocess.run(
            inp["command"], shell=True, capture_output=True, text=True, timeout=timeout_s,
        )
        output = result.stdout or ""
        if result.returncode != 0:
            stderr = f"\nStderr: {result.stderr}" if result.stderr else ""
            stdout = f"\nStdout: {result.stdout}" if result.stdout else ""
            return f"Command failed (exit code {result.returncode}){stdout}{stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {inp.get('timeout', 30000)}ms"
    except Exception as e:
        return f"Error: {e}"

#-------------------edit_file工具辅助--------------------------
def _find_actual_string(file_content: str, search_string: str) -> str | None:
    if search_string in file_content:
        return search_string
    norm_search = _normalize_quotes(search_string)
    norm_file = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        return file_content[idx:idx + len(search_string)]
    return None

def _normalize_quotes(s: str) -> str:
    s = re.sub("[‘’′]", "'", s)
    s = re.sub('[“”″]', '"', s)
    return s

def _generate_diff(old_content: str, old_string: str, new_string: str) -> str:
    before_change = old_content.split(old_string)[0]
    line_num = before_change.count("\n") + 1
    old_lines = old_string.split("\n")
    new_lines = new_string.split("\n")
    parts = [f"@@ -{line_num},{len(old_lines)} +{line_num},{len(new_lines)} @@"]
    for l in old_lines:
        parts.append(f"- {l}")
    for l in new_lines:
        parts.append(f"+ {l}")
    return "\n".join(parts)

#-------------------grep_search工具辅助------------------------
def _grep_python(pattern: str, directory: str, include: str | None) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"
    include_pattern = include
    matches: list[str] = []
    extra = 0

    def walk(d: str) -> None:
        nonlocal extra
        try:
            entries = os.listdir(d)
        except Exception:
            return
        for name in entries:
            if name.startswith(".") or name == "node_modules":
                continue
            full = os.path.join(d, name)
            if os.path.isdir(full):
                walk(full)
                continue
            if include_pattern and not fnmatch.fnmatch(name, include_pattern):
                continue
            try:
                text = Path(full).read_text(errors="replace")
                for i, line in enumerate(text.split("\n")):
                    if regex.search(line):
                        if len(matches) < 100:
                            matches.append(f"{full}:{i+1}:{line}")
                        else:
                            extra += 1
            except Exception:
                pass

    walk(directory)
    if not matches:
        return "No matches found."
    output = "\n".join(matches)
    if extra:
        output += f"\n... and {extra} more matches"
    return output

#------------------工具执行权限校验----------------------
_cached_json_rules: list[dict] | None = None

def check_permission(
    tool_name: str, inp: dict, mode: str = "default", plan_file_path: str | None = None,
) -> dict:
    """返回 {"action": "allow"|"deny"|"ask", "message": ...}。

    决策来源分两类：json 的 deny 规则 与 硬编码（内置兜底 + 动态模式）。
    按结果分类，每种 action 的所有情况：

    【deny】(3 种)
      ① json deny 规则命中 —— 最高优先级，连 bypassPermissions(yolo) 都拦
      ② plan 模式硬编码 —— 除写计划文件、派发子 agent 外，其余拒绝
      ③ 项目边界 —— 写工具目标在 {agent源码, cwd} 之外、或删除命令引用项目外路径

    【allow】(2 种)
      ① bypassPermissions 模式（yolo，跳过一切确认）
      ② 内置默认兜底 —— 非危险 shell 等（其余直接放行）

    【ask】(1 种)
      ① 内置默认兜底 —— 危险 shell、写文件(新或覆盖)、编辑文件
    """
    # ── deny ①：json deny 规则（最高，任何模式都拦）──
    if _check_json_rules(tool_name, inp):
        return {"action": "deny", "message": f"Denied by permission rule for {tool_name}"}

    # ── deny ②：plan 模式硬编码限制（只准写计划文件 + 派发子 agent + 白名单只读 MCP，其余拒绝）──
    if mode == "plan":
        writes_plan = (
            tool_name in EDIT_TOOLS
            and plan_file_path
            and (inp.get("file_path") or inp.get("path")) == plan_file_path
        ) #写文件路径与plan文件路径一致，，允许
        # MCP 工具：plan 模式只放行白名单内的只读工具（如联网搜索），其余拒绝
        allowed_mcp = tool_name in PLAN_ALLOWED_MCP_TOOLS
        if writes_plan or tool_name in {"agent", "memory_search"} or allowed_mcp:
            return {"action": "allow"}              # 派发只读子 agent 探索 / 白名单 MCP，允许
        return {"action": "deny", "message": f"Blocked in plan mode: {tool_name}"}

    # ── deny ③：项目边界（删除命令越界，硬性安全边界，连 yolo 都拦）──
    # 注：write_file/edit_file 的目录边界已取消（用户授权），可写任意目录
    if tool_name == "run_shell" and _delete_outside_project(inp.get("command", "")):
        return {"action": "deny", "message": "Delete command targets outside project directory"}

    # ── allow ①：bypassPermissions（yolo 模式跳过确认）──
    if mode == "bypassPermissions":
        return {"action": "allow"}

    # ── 内置默认（硬编码兜底，用户不写 settings 也顺畅）──
    if tool_name in {"memory_save", "memory_delete"}:
        return {"action": "ask", "message": f"{tool_name}: {json.dumps(inp, ensure_ascii=False)}"}
    # ask ②：危险 shell / 写文件(新或覆盖) / 编辑 → 询问
    if tool_name == "run_shell" and is_dangerous(inp.get("command", "")):
        return {"action": "ask", "message": inp.get("command", "")}
    if tool_name == "write_file":
        exists = Path(inp.get("file_path", "")).exists()
        verb = "overwrite file" if exists else "write new file"
        return {"action": "ask", "message": f"{verb}: {inp.get('file_path', '')}"}
    if tool_name == "edit_file":
        return {"action": "ask", "message": f"edit file: {inp.get('file_path', '')}"}
    # allow ③：其余（非危险 shell 等）直接放行
    return {"action": "allow"}

def _is_within_roots(target: str) -> bool:
    """判断目标路径是否在允许的根目录内（agent 源码 + cwd）。"""
    try:
        resolved = Path(target).resolve()
    except Exception:
        return False
    roots = [
        Path(__file__).resolve().parent.parent,  # agent 源码根目录
        Path.cwd().resolve(),                     # 当前项目
    ]
    #当前路径是否位于root内
    return any(resolved.is_relative_to(root) for root in roots)

def _delete_outside_project(command: str) -> bool:
    """近似判断删除命令是否引用了项目外路径（绝对路径、~/、..）。"""
    if not re.search(r"\b(?:rm|del|rmdir|Remove-Item)\b", command):
        return False
    return bool(re.search(r"(^|\s)(/|~/|[A-Za-z]:\\|\.\.)", command))

# runshelll中危险权限规则匹配的关键参数名
#
DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s"),
    re.compile(r"\bgit\s+(push|reset|clean|checkout\s+\.)"),
    re.compile(r"\bsudo\b"), re.compile(r"\bmkfs\b"), re.compile(r"\bdd\s"),
    re.compile(r">\s*/dev/"), re.compile(r"\bkill\b"), re.compile(r"\bpkill\b"),
    re.compile(r"\breboot\b"), re.compile(r"\bshutdown\b"),
    re.compile(r"\bdel\s", re.IGNORECASE), re.compile(r"\brmdir\s", re.IGNORECASE),
    re.compile(r"\bformat\s", re.IGNORECASE), re.compile(r"\btaskkill\s", re.IGNORECASE),
    re.compile(r"\bRemove-Item\s", re.IGNORECASE), re.compile(r"\bStop-Process\s", re.IGNORECASE),
]

def is_dangerous(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_PATTERNS)

#------------------settings.json权限-----------------------
def _check_json_rules(tool_name: str, inp: dict) -> bool:
    """查询 settings.json 的 deny 规则，命中返回 True。"""
    rules = _load_json_rules()
    return any(_matches_rule(rule, tool_name, inp) for rule in rules)

def _matches_rule(rule: dict, tool_name: str, inp: dict) -> bool:
    """判断一条规则是否命中某次工具调用。

    提取工具调用的关键参数值（见 _rule_match_value），
    用 glob 通配符（* ? [seq]）做匹配。
    """
    if rule["tool"] != tool_name:
        return False
    if rule["pattern"] is None:
        return True
    value = _rule_match_value(tool_name, inp)
    if not value:
        return False   # 无法提取匹配值（参数缺失），不匹配该规则，走后续权限检查
    return fnmatch.fnmatch(value, rule["pattern"])

def _load_json_rules() -> list[dict]:
    """读取 settings.json 的 deny 规则（合并用户级 + 项目级）。"""
    global _cached_json_rules
    if _cached_json_rules is not None:
        return _cached_json_rules
    deny: list[dict] = []
    user_settings = _load_settings(Path.home() / ".claude" / "settings.json")
    project_settings = _load_settings(Path.cwd() / ".claude" / "settings.json")
    for settings in [user_settings, project_settings]:
        if not settings or "permissions" not in settings:
            continue
        perms = settings["permissions"]
        for r in perms.get("deny", []):
            deny.append(_parse_rule(r))
    _cached_json_rules = deny
    return _cached_json_rules

def _parse_rule(rule: str) -> dict:
    m = re.match(r"^([a-z_]+)\((.+)\)$", rule)
    if m:
        return {"tool": m.group(1), "pattern": m.group(2)}
    return {"tool": rule, "pattern": None}

def _load_settings(file_path: Path) -> dict | None:
    if not file_path.exists():
        return None
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _rule_match_value(tool_name: str, inp: dict) -> str:
    """提取工具调用中用于权限规则匹配的关键参数值。

    不同工具匹配不同参数（run_shell→command、read_file→file_path、
    grep_search→pattern 等），见 _RULE_MATCH_PARAM。
    """
    param = _RULE_MATCH_PARAM.get(tool_name)
    return inp.get(param, "") if param else ""

_RULE_MATCH_PARAM = {
    "run_shell": "command",
    "read_file": "file_path",
    "write_file": "file_path",
    "edit_file": "file_path",
    "list_files": "pattern",
    "grep_search": "pattern",
 "skill": "skill_name",
}

    # ───防止工具返回值撑爆 LLM 上下文────────────────────

MAX_RESULT_CHARS = 50000

def truncate_result(result: str) -> str:
    if len(result) <= MAX_RESULT_CHARS:
        return result
    keep_each = (MAX_RESULT_CHARS - 60) // 2
    return (
        result[:keep_each]
        + f"\n\n[... truncated {len(result) - keep_each * 2} chars ...]\n\n"
        + result[-keep_each:]
    )

# ─── 执行工具调用────────────────

async def execute_tool(
    name: str, args: dict, read_file_state: dict[str, float] | None = None
) -> str:

    #线程执行读文件，并存mtime
    if name == "read_file":
        #获取文件内容
        result = await asyncio.to_thread(_read_file, args)
        #将已读的文件路径加入self._read_file_state,用于编辑文件前校验已读过
        if not result.startswith("Error"):
            abs_path = str(Path(args["file_path"]).resolve())
            try:
                read_file_state[abs_path] = os.path.getmtime(abs_path)
            except OSError:
                pass
        return result

    #编辑文件前检查：没读过 或 用户个人修改了文件，返回str告诉llm重新读文件再操作
    if name in ("write_file", "edit_file") and read_file_state is not None:
        abs_path = str(Path(args["file_path"]).resolve())
        if os.path.exists(abs_path):
            if abs_path not in read_file_state:
                verb = "writing" if name == "write_file" else "editing"
                return f"Error: You must read this file before {verb}. Use read_file first."
            if os.path.getmtime(abs_path) != read_file_state[abs_path]:
                verb = "writing" if name == "write_file" else "editing"
                return f"Warning: {args['file_path']} was modified externally. Please read_file again before {verb}."

    #在线程执行工具调用
    handlers: dict = {
        "write_file": _write_file, "edit_file": _edit_file,
        "list_files": _list_files, "grep_search": _grep_search,
        "run_shell": _run_shell,
        "skill": _execute_skill,
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    result = await asyncio.to_thread(handler, args)

    #编辑文件后，存入最后遍及时间
    if name in ("write_file", "edit_file") and read_file_state is not None and not result.startswith("Error"):
        #resolve为解析成绝对路径
        abs_path = str(Path(args["file_path"]).resolve())
        try:
            #read_file_state存入文件编辑的完成时间
            read_file_state[abs_path] = os.path.getmtime(abs_path)
        except OSError:
            pass

    return result

