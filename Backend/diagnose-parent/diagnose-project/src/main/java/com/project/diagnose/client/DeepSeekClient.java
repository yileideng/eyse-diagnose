package com.project.diagnose.client;

import com.alibaba.fastjson.JSON;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.diagnose.dto.request.DeepSeekRequestModel;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;

@Component
public class DeepSeekClient {
    /**
     * 请求API地址
     */
    @Value("${deepseek.api.url}")
    private String API_URL;
    /**
     * 你在DeepSeek官网申请的API KEY，注意不要泄露给他人！
     */
    @Value("${deepseek.api.key}")
    private String API_KEY;

    @Autowired
    private OkHttpClient client;


    @Async
    public CompletableFuture<String> getResponse(DeepSeekRequestModel requestModel) {
        String jsonBody = JSON.toJSONString(requestModel);

        Request request = new Request.Builder()
                .url(API_URL)
                .post(RequestBody.create(jsonBody, MediaType.get("application/json")))
                .addHeader("Authorization", "Bearer " + API_KEY)
                .build();

        CompletableFuture<String> future = new CompletableFuture<>();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                future.completeExceptionally(e);
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful() && response.body() != null) {
                    String jsonResponse = response.body().string();
                    try {
                        JsonNode root = new ObjectMapper().readTree(jsonResponse);
                        String answer = root.get("choices").get(0).get("message").get("content").asText();
                        future.complete(answer);
                    } catch (IOException e) {
                        future.completeExceptionally(e);
                    }
                } else {
                    future.completeExceptionally(new IOException("Unexpected code " + response));
                }
            }
        });

        return future;
    }
}



