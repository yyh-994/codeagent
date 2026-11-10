FROM python:3.12-slim

# 安装 Git（Agent 需要调 git 命令）
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 复制项目代码
COPY . /app/claude-code-agent

# 安装 Agent
RUN pip install --no-cache-dir -e /app/claude-code-agent

# 用户的项目代码通过 volume 挂载到 /workspace
# API Key 通过环境变量传入

ENTRYPOINT ["claude-code-agent"]
