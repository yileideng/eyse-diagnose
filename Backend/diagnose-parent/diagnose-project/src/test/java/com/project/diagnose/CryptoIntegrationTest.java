package com.project.diagnose;

import com.project.diagnose.service.CryptoService;
import com.project.diagnose.service.KeyManagementService;
import com.project.diagnose.dto.response.CryptoResponse;
import com.project.diagnose.dto.response.KeyPairResponse;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 加密集成测试
 */
@SpringBootTest
public class CryptoIntegrationTest {
    
    @Autowired
    private CryptoService cryptoService;
    
    @Autowired
    private KeyManagementService keyManagementService;
    
    @Test
    public void testGenerateUserKeys() {
        KeyPairResponse keyPair = cryptoService.generateUserKeys();
        
        assertNotNull(keyPair);
        assertNotNull(keyPair.getPrivateKey());
        assertNotNull(keyPair.getPublicKey());
        assertTrue(keyPair.getPrivateKey().contains("BEGIN PRIVATE KEY"));
        assertTrue(keyPair.getPublicKey().contains("BEGIN PUBLIC KEY"));
    }
    
    @Test
    public void testGenerateModelKeys() {
        KeyPairResponse keyPair = cryptoService.generateModelKeys();
        
        assertNotNull(keyPair);
        assertNotNull(keyPair.getPrivateKey());
        assertNotNull(keyPair.getPublicKey());
        assertTrue(keyPair.getPrivateKey().contains("BEGIN PRIVATE KEY"));
        assertTrue(keyPair.getPublicKey().contains("BEGIN PUBLIC KEY"));
    }
    
    @Test
    public void testEncryptImage() throws IOException {
        // 生成密钥对
        KeyPairResponse userKeys = cryptoService.generateUserKeys();
        KeyPairResponse modelKeys = cryptoService.generateModelKeys();
        
        // 创建测试图片文件
        byte[] imageData = "test image data".getBytes();
        MultipartFile mockFile = new MockMultipartFile(
            "image", 
            "test.jpg", 
            "image/jpeg", 
            imageData
        );
        
        // 加密图片
        CryptoResponse response = cryptoService.encryptImage(
            mockFile, 
            userKeys.getPrivateKey(), 
            modelKeys.getPublicKey()
        );
        
        assertNotNull(response);
        assertEquals("success", response.getStatus());
        assertNotNull(response.getData());
        assertNotNull(response.getData().getMesData());
        assertNotNull(response.getData().getEncryptedImage());
        assertNotNull(response.getData().getSignature());
        assertNotNull(response.getData().getEncryptedAesKey());
    }
    
    @Test
    public void testKeyManagementService() {
        // 测试初始化模型密钥
        keyManagementService.initializeModelKeys();
        
        // 获取模型公钥
        String modelPublicKey = keyManagementService.getModelPublicKey();
        assertNotNull(modelPublicKey);
        assertTrue(modelPublicKey.contains("BEGIN PUBLIC KEY"));
    }
}
