# codeagent

基于 LangGraph 的 Python 编程智能体，支持 Anthropic 和 OpenAI 兼容接口。

## 目录

- `agent/`：主要项目代码、测试、Docker 配置与使用文档。
- `deepagents-main/`：项目目录中保留的 Deep Agents 参考源码，其许可证见对应目录。

## 快速开始

```powershell
cd agent
python -m pip install -e .
Copy-Item .env.example .env
# 编辑 .env，填写自己的 API 密钥
claude-code-agent
```

更多配置和功能说明见 [agent/README.md](agent/README.md)。

本仓库不包含本地密钥、环境变量文件、会话记录、数据库、虚拟环境或项目压缩包。
