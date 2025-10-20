# Eyse Diagnose 项目

## 项目简介
本仓库为眼科诊断系统的代码集合，包含后端（Java Spring）、前端（静态页面 + JS）、以及机器学习相关模块（Python）。

目录概要：

- Backend/: 后端 Java 项目与部署说明
  - diagnose-parent/diagnose-project: 主 Spring Boot 应用
- Frontend/: 前端静态页面与 API 调用脚本
- ML/: 机器学习模型、数据处理与推理管线
- Crypt0model/: 与加密相关的 Python 脚本与工具

## 快速使用说明
后端（Java）:
1. 进入 `Backend/diagnose-parent/diagnose-project` 目录
2. 使用 Maven 构建：
   mvn clean package
3. 在 `target/` 中获取 jar 并运行：
   java -jar target/diagnose-project-1.0-SNAPSHOT.jar

前端:
- 直接打开 `Frontend/` 下的 HTML 文件或用静态服务器（如 `http-server`）托管。

ML:
- 进入 `ML/` 目录查看 README 与 `pip` 依赖说明（本仓库未包含完整虚拟环境）。

Crypt0model:
- 包含部分脚本用于生成密钥与演示加密流程，仅作参考，使用前请审计安全性。

