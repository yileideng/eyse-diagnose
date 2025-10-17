# 加密模块集成说明

## 概述

本项目已成功集成加密模块，实现了用户图片的端到端加密传输。加密流程遵循以下步骤：

1. 用户上传图片到后端
2. 后端调用加密模块对图片进行加密
3. 将加密后的MES数据发送给机器学习端
4. 机器学习端解密并处理图片

## 加密流程详解

### 1. 密钥管理
- **用户密钥对**：每个用户都有独立的RSA密钥对（私钥+公钥）
- **模型密钥对**：系统有一个统一的模型密钥对
- **密钥存储**：用户私钥存储在数据库中，模型公钥存储在Redis中

### 2. 图片加密流程
1. 生成AES密钥
2. 使用AES密钥加密图片
3. 计算加密后图片的SHA-256哈希值
4. 使用用户私钥对哈希值进行RSA签名
5. 使用模型公钥对AES密钥进行RSA加密
6. 将加密图片、签名、加密密钥打包成MES格式

### 3. 数据传输
- 原始图片不再直接发送给机器学习端
- 发送的是MES格式的加密数据包
- 机器学习端需要相应的解密逻辑来处理MES数据

## 新增组件

### 1. 服务类
- `CryptoService`: 加密服务接口
- `CryptoServiceImpl`: 加密服务实现
- `KeyManagementService`: 密钥管理服务接口
- `KeyManagementServiceImpl`: 密钥管理服务实现

### 2. DTO类
- `CryptoRequest`: 加密请求DTO
- `CryptoResponse`: 加密响应DTO
- `KeyPairResponse`: 密钥对响应DTO

### 3. 控制器
- `CryptoController`: 加密相关API控制器

### 4. 数据库变更
- `User`表新增字段：
  - `private_key`: 用户私钥
  - `public_key`: 用户公钥

## API接口

### 1. 生成用户密钥对
```
POST /crypto/generate-user-keys
Headers: Authorization: Bearer <token>
Response: {
  "code": 200,
  "data": {
    "privateKey": "-----BEGIN PRIVATE KEY-----...",
    "publicKey": "-----BEGIN PUBLIC KEY-----..."
  }
}
```

### 2. 初始化模型密钥对
```
POST /crypto/initialize-model-keys
Response: {
  "code": 200,
  "message": "模型密钥对初始化成功"
}
```

## 配置说明

### application.yml配置
```yaml
# 加密服务配置
crypto:
  service:
    url: http://localhost:8080  # 加密模块服务地址
  model:
    key:
      redis:
        key: model:public:key   # Redis中模型公钥的键名
```

## 使用流程

### 1. 系统初始化
1. 启动加密模块服务（端口8080）
2. 调用 `/crypto/initialize-model-keys` 初始化模型密钥对

### 2. 用户注册/登录
1. 用户注册或首次登录时，系统会自动生成用户密钥对
2. 密钥对存储在用户表中

### 3. 图片诊断
1. 用户上传图片
2. 系统自动获取用户密钥和模型公钥
3. 调用加密模块对图片进行加密
4. 将加密后的MES数据发送给机器学习端
5. 机器学习端处理并返回诊断结果

## 注意事项

### 1. 加密模块依赖
- 需要确保加密模块服务（Python Flask应用）正在运行
- 默认端口：8080
- 需要安装Python依赖：cryptography, flask等

### 2. 机器学习端适配
- 机器学习端需要添加解密逻辑来处理MES数据
- 需要实现相应的解密接口

### 3. 密钥安全
- 用户私钥存储在数据库中，建议加密存储
- 模型私钥需要安全保管，不能泄露

### 4. 性能考虑
- 加密过程会增加处理时间
- 建议对加密操作进行性能监控
- 可以考虑异步处理加密任务

## 测试

运行测试类 `CryptoIntegrationTest` 来验证加密功能：

```bash
mvn test -Dtest=CryptoIntegrationTest
```

## 故障排除

### 1. Redis配置问题
**错误信息**: `No qualifying bean of type 'org.springframework.data.redis.core.RedisTemplate' available`

**解决方案**: 
- 确保项目中已添加`spring-boot-starter-data-redis`依赖
- 检查`application.yml`中的Redis配置是否正确
- 确保Redis服务器正在运行

**配置示例**:
```yaml
spring:
  redis:
    host: localhost
    port: 6379
    timeout: 2000ms
```

### 2. 加密模块连接失败
- 检查加密模块服务是否启动
- 检查配置中的服务地址是否正确
- 检查网络连接

### 3. 密钥生成失败
- 检查Redis连接
- 检查数据库连接
- 查看日志中的详细错误信息

### 4. 图片加密失败
- 检查图片文件是否有效
- 检查密钥是否正确
- 查看加密模块的日志

## 未来优化

1. **密钥轮换**：定期更新用户和模型密钥
2. **缓存优化**：缓存模型公钥减少Redis访问
3. **异步处理**：将加密操作改为异步处理
4. **监控告警**：添加加密操作的监控和告警
5. **密钥备份**：实现密钥的备份和恢复机制
