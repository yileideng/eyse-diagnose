package com.project.diagnose.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.diagnose.dto.response.DiagnoseResponse;
import com.project.diagnose.exception.DiagnoseException;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import okio.BufferedSink;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.List;

@Slf4j
@Service
public class MLClient {

    private final String mlApiUrl = "http://localhost:8082/mock/files";
    @Autowired
    private OkHttpClient client;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public DiagnoseResponse requestForDiagnose(List<File> files) throws IOException {
        /*if (files == null || files.size() == 0) {
            throw new DiagnoseException("上传的诊断图片不能为空");
        }
        // 创建一个 MultipartBody.Builder
        MultipartBody.Builder requestBodyBuilder = new MultipartBody.Builder();
        requestBodyBuilder.setType(MultipartBody.FORM);

        // 添加每个文件到请求体
        for (File file : files) {
            // 创建文件请求体部分
            RequestBody fileBody = RequestBody.create(
                    MediaType.get("image/*"), // 根据文件类型设置 MIME 类型
                    file
            );
            // 添加文件到表单字段 "fileList"
            requestBodyBuilder.addPart(
                    Headers.of("Content-Disposition", "form-data; name=\"fileList\"; filename=\"" + file.getName() + "\""),
                    fileBody
            );
        }

        // 构建请求体
        RequestBody requestBody = requestBodyBuilder.build();*/

        // 创建请求
        Request request = new Request.Builder()
                .url("http://localhost:8082/mock/files") // 替换为你的服务端地址
                .post(new RequestBody() {
                    @Nullable
                    @Override
                    public MediaType contentType() {
                        return null;
                    }

                    @Override
                    public void writeTo(@NotNull BufferedSink bufferedSink) throws IOException {

                    }
                })
                .build();

        DiagnoseResponse diagnoseResponse = null;
        // 发送请求并处理响应
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() && response.body() != null) {
                String responseBody = response.body().string();
               log.info("响应数据: {}", responseBody);
                // 处理返回的 JSON 数据
                diagnoseResponse = objectMapper.readValue(responseBody, DiagnoseResponse.class);
                log.info(diagnoseResponse.toString());
            } else {
                System.out.println("请求失败: " + response);
            }
        }
        return diagnoseResponse;
    }
}
