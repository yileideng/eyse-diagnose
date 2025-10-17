# 应用程序启动指南

## 前置条件

### 1. 环境要求
- Java 11 或更高版本
- Maven 3.6 或更高版本
- Redis 服务器
- MySQL 数据库
- Python 3.7+ (用于加密模块)

### 2. 服务依赖
- **Redis服务器**: 端口 6379
- **MySQL数据库**: 端口 3306
- **加密模块服务**: 端口 8080

## 启动步骤

### 1. 启动Redis服务器
```bash
# Windows
redis-server

# Linux/Mac
sudo systemctl start redis
# 或者
redis-server
```

### 2. 启动MySQL数据库
```bash
# Windows
net start mysql

# Linux/Mac
sudo systemctl start mysql
# 或者
sudo service mysql start
```

### 3. 启动加密模块服务
```bash
cd Crypt0model
python api/encrypt_api.py
python api/genkey_api.py
```

### 4. 启动后端应用程序
```bash
cd Backend/diagnose-parent/diagnose-project
mvn spring-boot:run
```

## 验证启动

### 1. 检查应用程序状态
访问: http://localhost:8082/actuator/health

### 2. 初始化模型密钥
```bash
curl -X POST http://localhost:8082/crypto/initialize-model-keys
```

### 3. 测试加密功能
```bash
# 生成用户密钥对
curl -X POST http://localhost:8082/crypto/generate-user-keys \
  -H "Authorization: Bearer <your-token>"
```

## 常见问题

### 1. Redis连接失败
**错误**: `Connection refused: connect`

**解决方案**:
- 检查Redis是否启动: `redis-cli ping`
- 检查端口是否被占用: `netstat -an | findstr 6379`
- 检查防火墙设置

### 2. 数据库连接失败
**错误**: `Access denied for user 'root'@'localhost'`

**解决方案**:
- 检查数据库用户名和密码
- 确保数据库服务正在运行
- 检查数据库连接配置

### 3. 加密模块连接失败
**错误**: `Connection refused: connect`

**解决方案**:
- 检查加密模块是否启动
- 检查端口8080是否被占用
- 确保Python依赖已安装

### 4. 端口冲突
**错误**: `Port 8082 was already in use`

**解决方案**:
- 更改应用程序端口: 修改`application.yml`中的`server.port`
- 或者停止占用端口的进程

## 开发环境配置

### 1. IDE配置
- 推荐使用IntelliJ IDEA或Eclipse
- 确保Maven项目正确导入
- 配置正确的JDK版本

### 2. 调试配置
```yaml
# application-dev.yml
logging:
  level:
    com.project.diagnose: DEBUG
    org.springframework.data.redis: DEBUG
```

### 3. 测试环境
```bash
# 运行测试
mvn test

# 运行特定测试
mvn test -Dtest=CryptoIntegrationTest
```

## 生产环境部署

### 1. 构建应用
```bash
mvn clean package -DskipTests
```

### 2. 运行JAR文件
```bash
java -jar target/diagnose-project-1.0-SNAPSHOT.jar
```

### 3. 使用Docker部署
```dockerfile
FROM openjdk:11-jre-slim
COPY target/diagnose-project-1.0-SNAPSHOT.jar app.jar
EXPOSE 8082
ENTRYPOINT ["java", "-jar", "/app.jar"]
```

## 监控和日志

### 1. 应用监控
- 健康检查: `/actuator/health`
- 指标监控: `/actuator/metrics`
- 应用信息: `/actuator/info`

### 2. 日志配置
```yaml
logging:
  level:
    root: INFO
    com.project.diagnose: DEBUG
  file:
    name: logs/application.log
```

### 3. 性能监控
- 使用Micrometer进行指标收集
- 集成Prometheus和Grafana
- 监控Redis和数据库性能
