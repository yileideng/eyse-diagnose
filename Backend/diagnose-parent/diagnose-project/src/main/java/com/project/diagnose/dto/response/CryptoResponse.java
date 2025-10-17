package com.project.diagnose.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

/**
 * 加密响应DTO
 */
@Data
public class CryptoResponse {
    /**
     * 加密状态
     */
    private String status;
    
    /**
     * 响应消息
     */
    private String message;
    
    /**
     * 加密数据
     */
    private CryptoData data;
    
    @Data
    public static class CryptoData {
        /**
         * MES数据的base64编码
         */
        @JsonProperty("mes_data")
        private String mesData;
        
        /**
         * 加密后的图片
         */
        @JsonProperty("encrypted_image")
        private String encryptedImage;
        
        /**
         * 签名
         */
        private String signature;
        
        /**
         * 加密的AES密钥
         */
        @JsonProperty("encrypted_aes_key")
        private String encryptedAesKey;
        
        /**
         * 原始AES密钥（仅用于测试）
         */
        @JsonProperty("aes_key")
        private String aesKey;
        
        /**
         * MES数据大小
         */
        @JsonProperty("mes_size")
        private Integer mesSize;
    }
}
