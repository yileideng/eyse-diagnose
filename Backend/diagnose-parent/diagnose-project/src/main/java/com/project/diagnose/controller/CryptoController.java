package com.project.diagnose.controller;

import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.dto.response.KeyPairResponse;
import com.project.diagnose.dto.vo.Result;
import com.project.diagnose.service.KeyManagementService;
import com.project.diagnose.utils.RedisUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

/**
 * 加密相关控制器
 */
@Slf4j
@RestController
@RequestMapping("/crypto")
public class CryptoController {
    
    @Autowired
    private KeyManagementService keyManagementService;
    
    @Autowired
    private RedisUtils redisUtils;
    
    /**
     * 为用户生成密钥对
     */
    @PostMapping("/generate-user-keys")
    @LogAnnotation(module = "CryptoController", operator = "生成用户密钥对")
    public Result<KeyPairResponse> generateUserKeys(@RequestHeader("Authorization") String token) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        KeyPairResponse keyPair = keyManagementService.generateUserKeys(userId);
        return Result.success(keyPair);
    }
    
    /**
     * 初始化模型密钥对
     */
    @PostMapping("/initialize-model-keys")
    @LogAnnotation(module = "CryptoController", operator = "初始化模型密钥对")
    public Result<String> initializeModelKeys() {
        keyManagementService.initializeModelKeys();
        return Result.success("模型密钥对初始化成功");
    }
}
