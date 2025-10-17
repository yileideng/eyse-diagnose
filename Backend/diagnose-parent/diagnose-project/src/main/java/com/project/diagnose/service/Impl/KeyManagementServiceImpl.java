package com.project.diagnose.service.Impl;

import com.project.diagnose.dto.response.KeyPairResponse;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.User;
import com.project.diagnose.service.CryptoService;
import com.project.diagnose.service.KeyManagementService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

/**
 * 密钥管理服务实现类
 */
@Slf4j
@Service
public class KeyManagementServiceImpl implements KeyManagementService {
    
    @Autowired
    private CryptoService cryptoService;
    
    @Autowired
    private UserMapper userMapper;
    
    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    
    @Value("${crypto.model.key.redis.key:model:public:key}")
    private String modelPublicKeyRedisKey;
    
    @Override
    public KeyPairResponse generateUserKeys(Long userId) {
        try {
            // 生成密钥对
            KeyPairResponse keyPair = cryptoService.generateUserKeys();
            
            // 更新用户密钥信息
            User user = new User();
            user.setId(userId);
            user.setPrivateKey(keyPair.getPrivateKey());
            user.setPublicKey(keyPair.getPublicKey());
            
            int result = userMapper.updateById(user);
            if (result <= 0) {
                throw new DiagnoseException("更新用户密钥失败");
            }
            
            log.info("为用户 {} 生成密钥对成功", userId);
            return keyPair;
        } catch (Exception e) {
            log.error("为用户 {} 生成密钥对失败", userId, e);
            throw new DiagnoseException("生成用户密钥失败: " + e.getMessage());
        }
    }
    
    @Override
    public User getUserWithKeys(Long userId) {
        User user = userMapper.selectById(userId);
        if (user == null) {
            throw new DiagnoseException("用户不存在");
        }
        
        // 如果用户没有密钥，自动生成
        if (user.getPrivateKey() == null || user.getPublicKey() == null) {
            log.info("用户 {} 没有密钥，自动生成", userId);
            generateUserKeys(userId);
            // 重新查询用户信息
            user = userMapper.selectById(userId);
        }
        
        return user;
    }
    
    @Override
    public String getModelPublicKey() {
        // 先从Redis缓存中获取
        String modelPublicKey = redisTemplate.opsForValue().get(modelPublicKeyRedisKey);
        
        if (modelPublicKey == null) {
            // 如果缓存中没有，初始化模型密钥
            initializeModelKeys();
            modelPublicKey = redisTemplate.opsForValue().get(modelPublicKeyRedisKey);
        }
        
        return modelPublicKey;
    }
    
    @Override
    public void initializeModelKeys() {
        try {
            // 检查Redis中是否已有模型公钥
            String existingKey = redisTemplate.opsForValue().get(modelPublicKeyRedisKey);
            if (existingKey != null) {
                log.info("模型公钥已存在，无需重新生成");
                return;
            }
            
            // 生成模型密钥对
            KeyPairResponse modelKeyPair = cryptoService.generateModelKeys();
            
            // 将模型公钥存储到Redis，设置过期时间为30天
            redisTemplate.opsForValue().set(
                modelPublicKeyRedisKey, 
                modelKeyPair.getPublicKey(), 
                30, 
                TimeUnit.DAYS
            );
            
            log.info("模型密钥对初始化成功");
        } catch (Exception e) {
            log.error("初始化模型密钥失败", e);
            throw new DiagnoseException("初始化模型密钥失败: " + e.getMessage());
        }
    }
}
