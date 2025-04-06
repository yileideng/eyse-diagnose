package com.project.diagnose.service.Impl;

import com.project.diagnose.client.DeepSeekClient;
import com.project.diagnose.dto.request.DeepSeekRequestModel;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.service.DeepSeekService;
import com.project.diagnose.utils.RedisUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

@Slf4j
@Service
public class DeepSeekServiceImpl implements DeepSeekService {
    @Autowired
    private DeepSeekClient deepSeekClient;

    @Override
    public String interact(String prompt) {

        // 构建请求消息
        DeepSeekRequestModel.Message requestMessage = new DeepSeekRequestModel.Message("user", prompt);
        //requestModel.setStream(true);

        DeepSeekRequestModel requestModel = new DeepSeekRequestModel("deepseek-chat", requestMessage);


        // 发送请求与大模型交互
        CompletableFuture<String> future = deepSeekClient.getResponse(requestModel);

        try {
            // 获取请求结果
            String answer = future.get();

            return answer;
        } catch (InterruptedException | ExecutionException e) {
            log.info("获取deepseek交互的回调结果失败: {}", e.getMessage());
            throw new DiagnoseException("获取deepseek交互的回调结果失败");
        }
    }
}
