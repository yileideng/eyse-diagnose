package com.project.diagnose.service;

import com.project.diagnose.dto.response.CryptoResponse;
import com.project.diagnose.dto.response.KeyPairResponse;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;

/**
 * 加密服务接口
 */
public interface CryptoService {
    
    /**
     * 生成用户密钥对
     * @return 密钥对
     */
    KeyPairResponse generateUserKeys();
    
    /**
     * 生成模型密钥对
     * @return 密钥对
     */
    KeyPairResponse generateModelKeys();
    
    /**
     * 加密图片文件
     * @param imageFile 图片文件
     * @param userPrivateKey 用户私钥
     * @param modelPublicKey 模型公钥
     * @return 加密结果
     */
    CryptoResponse encryptImage(MultipartFile imageFile, String userPrivateKey, String modelPublicKey);
    
    /**
     * 加密图片文件（通过文件路径）
     * @param imageFile 图片文件
     * @param userPrivateKey 用户私钥
     * @param modelPublicKey 模型公钥
     * @return 加密结果
     */
    CryptoResponse encryptImage(File imageFile, String userPrivateKey, String modelPublicKey);
}
