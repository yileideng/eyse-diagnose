package com.project.diagnose.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

/**
 * 密钥对响应DTO
 */
@Data
public class KeyPairResponse {
    /**
     * 私钥
     */
    @JsonProperty("private_key")
    private String privateKey;
    
    /**
     * 公钥
     */
    @JsonProperty("public_key")
    private String publicKey;
}
