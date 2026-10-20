# 催收实时对话助手 Docker 部署指南

本项目通过 `docker-compose` 编排了应用服务 (`app`)、向量数据库 (`db`) 和缓存 (`redis`)。
支持**挂载本地 `.env` 文件动态覆盖容器配置**。

## 一、准备工作

1. 确保宿主机已安装 `Docker` 和 `Docker Compose`。
2. 将原始录音数据放入宿主机项目的 `data/calls.json`。
3. **创建并配置本地 `.env`** （核心步骤）：
   ```bash
   # 在项目根目录（与 docker 文件夹平级）执行：
   cp .env.example .env
   
   # 编辑 .env，填入真实的 LLM_API_KEY
   # 注意：请务必确保 USE_EMBEDDED_DB=false, USE_EMBEDDED_REDIS=false