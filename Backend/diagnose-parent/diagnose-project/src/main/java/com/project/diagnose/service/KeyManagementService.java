package com.project.diagnose.service;

import com.project.diagnose.dto.response.KeyPairResponse;
import com.project.diagnose.pojo.User;

/**
 * 密钥管理服务接口
 */
public interface KeyManagementService {
    
    /**
     * 为用户生成密钥对
     * @param userId 用户ID
     * @return 密钥对
     */
    KeyPairResponse generateUserKeys(Long userId);
    
    /**
     * 获取用户密钥对
     * @param userId 用户ID
     * @return 用户信息（包含密钥）
     */
    User getUserWithKeys(Long userId);
    
    /**
     * 获取模型公钥
     * @return 模型公钥
     */
    String getModelPublicKey();
    
    /**
     * 初始化模型密钥对（如果不存在）
     */
    void initializeModelKeys();
}
