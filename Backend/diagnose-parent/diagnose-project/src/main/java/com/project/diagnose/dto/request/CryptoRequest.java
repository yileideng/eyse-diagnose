package com.project.diagnose.dto.request;

import lombok.Data;

/**
 * 加密请求DTO
 */
@Data
public class CryptoRequest {
    /**
     * 用户私钥
     */
    private String userPrivateKey;
    
    /**
     * 模型公钥
     */
    private String modelPublicKey;
    
    /**
     * 图片文件路径或URL
     */
    private String imagePath;
}
