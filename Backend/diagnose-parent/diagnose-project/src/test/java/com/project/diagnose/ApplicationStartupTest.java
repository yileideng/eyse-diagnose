package com.project.diagnose;

import com.project.diagnose.service.CryptoService;
import com.project.diagnose.service.KeyManagementService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * 应用程序启动测试
 * 验证所有Bean都能正确注入
 */
@SpringBootTest
@ActiveProfiles("test")
public class ApplicationStartupTest {
    
    @Autowired
    private CryptoService cryptoService;
    
    @Autowired
    private KeyManagementService keyManagementService;
    
    @Test
    public void testApplicationContextLoads() {
        // 验证应用程序上下文能够正确加载
        assertNotNull(cryptoService, "CryptoService should be injected");
        assertNotNull(keyManagementService, "KeyManagementService should be injected");
    }
    
    @Test
    public void testServicesAreAvailable() {
        // 验证服务类能够正确实例化
        assertNotNull(cryptoService, "CryptoService should not be null");
        assertNotNull(keyManagementService, "KeyManagementService should not be null");
    }
}
