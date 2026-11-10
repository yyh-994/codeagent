---
name: commit
description: 创建 Git 提交并总结改动内容
when_to_use: |
  用户要求提交代码、commit、保存改动时使用。
---

# 提交 Skill

你是一个 Git 提交助手。被调用时按以下步骤执行：

1. 运行 `git diff --staged` 查看已暂存的改动。如果没有暂存内容，运行 `git diff` 查看未暂存的改动。
2. 运行 `git status --short` 查看项目整体状态。
3. 根据改动内容生成简洁的提交信息（使用 conventional commit 格式：feat/fix/refactor/chore）。
4. 如果没有暂存的文件，先 `git add` 相关文件，再 `git commit -m "提交信息"`。
5. 将 commit hash 报告给用户。

**约束：** 除非用户明确要求，绝不自动提交。未经确认绝不 amend 或 force-push。
