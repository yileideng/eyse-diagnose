package com.project.diagnose.service.Impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.diagnose.dto.response.CryptoResponse;
import com.project.diagnose.dto.response.KeyPairResponse;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.service.CryptoService;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

/**
 * 加密服务实现类
 */
@Slf4j
@Service
public class CryptoServiceImpl implements CryptoService {

    private String generateKeysUrl = "http://localhost:8082";

    private String encryptImageUrl = "http://localhost:8080";
    
    @Autowired
    private OkHttpClient okHttpClient;
    
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    @Override
    public KeyPairResponse generateUserKeys() {
        return generateKeys("/generate-user-keys");
    }
    
    @Override
    public KeyPairResponse generateModelKeys() {
        return generateKeys("/generate-model-keys");
    }
    
    private KeyPairResponse generateKeys(String endpoint) {
        try {
            Request request = new Request.Builder()
                    .url(generateKeysUrl + endpoint)
                    .get()
                    .build();
            
            try (Response response = okHttpClient.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    String responseBody = response.body().string();
                    log.info("密钥生成响应: {}", responseBody);
                    return objectMapper.readValue(responseBody, KeyPairResponse.class);
                } else {
                    throw new DiagnoseException("生成密钥失败: " + response.code());
                }
            }
        } catch (IOException e) {
            log.error("生成密钥时发生错误", e);
            throw new DiagnoseException("生成密钥失败: " + e.getMessage());
        }
    }
    
    @Override
    public CryptoResponse encryptImage(MultipartFile imageFile, String userPrivateKey, String modelPublicKey) {
        try {
            // 创建请求体
            MultipartBody.Builder requestBodyBuilder = new MultipartBody.Builder();
            requestBodyBuilder.setType(MultipartBody.FORM);
            
            // 添加图片文件
            RequestBody fileBody = RequestBody.create(
                    MediaType.parse("image/*"),
                    imageFile.getBytes()
            );
            requestBodyBuilder.addFormDataPart("image", imageFile.getOriginalFilename(), fileBody);
            
            // 添加密钥
            requestBodyBuilder.addFormDataPart("user_private_key", userPrivateKey);
            requestBodyBuilder.addFormDataPart("model_public_key", modelPublicKey);
            
            RequestBody requestBody = requestBodyBuilder.build();
            
            // 创建请求
            Request request = new Request.Builder()
                    .url(encryptImageUrl + "/encrypt-image")
                    .post(requestBody)
                    .build();
            
            // 发送请求
            try (Response response = okHttpClient.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    String responseBody = response.body().string();
                    log.info("图片加密响应: {}", responseBody);
                    return objectMapper.readValue(responseBody, CryptoResponse.class);
                } else {
                    throw new DiagnoseException("图片加密失败: " + response.code());
                }
            }
        } catch (IOException e) {
            log.error("图片加密时发生错误", e);
            throw new DiagnoseException("图片加密失败: " + e.getMessage());
        }
    }
    
    @Override
    public CryptoResponse encryptImage(File imageFile, String userPrivateKey, String modelPublicKey) {
        try {
            // 创建请求体
            MultipartBody.Builder requestBodyBuilder = new MultipartBody.Builder();
            requestBodyBuilder.setType(MultipartBody.FORM);
            
            // 添加图片文件
            RequestBody fileBody = RequestBody.create(
                    MediaType.parse("image/*"),
                    imageFile
            );
            requestBodyBuilder.addFormDataPart("image", imageFile.getName(), fileBody);
            
            // 添加密钥
            requestBodyBuilder.addFormDataPart("user_private_key", userPrivateKey);
            requestBodyBuilder.addFormDataPart("model_public_key", modelPublicKey);
            
            RequestBody requestBody = requestBodyBuilder.build();
            
            // 创建请求
            Request request = new Request.Builder()
                    .url(encryptImageUrl + "/encrypt-image")
                    .post(requestBody)
                    .build();
            
            // 发送请求
            try (Response response = okHttpClient.newCall(request).execute()) {
                if (response.isSuccessful() && response.body() != null) {
                    String responseBody = response.body().string();
                    return objectMapper.readValue(responseBody, CryptoResponse.class);
                } else {
                    throw new DiagnoseException("图片加密失败: " + response.code());
                }
            }
        } catch (IOException e) {
            log.error("图片加密时发生错误", e);
            throw new DiagnoseException("图片加密失败: " + e.getMessage());
        }
    }
}
