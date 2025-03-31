package com.project.diagnose.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.diagnose.dto.response.BulkDiagnoseResponse;
import com.project.diagnose.dto.response.PersonalDiagnoseResponse;
import com.project.diagnose.exception.DiagnoseException;
import lombok.extern.slf4j.Slf4j;
import okhttp3.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.List;

@Slf4j
@Service
public class MLClient {

    private final String baseUrl = "http://localhost:8082";
    @Autowired
    private OkHttpClient client;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public BulkDiagnoseResponse requestForBulkDiagnose(List<File> files) throws IOException {
        String methodUrl = "/mock/file-bulk";
        if (files == null || files.size() == 0) {
            throw new DiagnoseException("上传的诊断图片不能为空");
        }
        // 创建一个 MultipartBody.Builder
        MultipartBody.Builder requestBodyBuilder = new MultipartBody.Builder();
        requestBodyBuilder.setType(MultipartBody.FORM);

        // 添加每个文件到请求体
        for (File file : files) {
            // 创建文件请求体部分
            RequestBody fileBody = RequestBody.create(
                    //MediaType.get("image/*"), // 根据文件类型设置 MIME 类型
                    MediaType.get("application/zip"), // 根据文件类型设置 MIME 类型
                    file
            );
            // 添加文件到表单字段 "fileList"
            requestBodyBuilder.addPart(
                    Headers.of("Content-Disposition", "form-data; name=\"file\"; filename=\"" + file.getName() + "\""),
                    fileBody
            );
        }

        // 构建请求体
        RequestBody requestBody = requestBodyBuilder.build();

        // 创建请求
        Request request = new Request.Builder()
                .url(baseUrl + methodUrl) // 替换为你的服务端地址
                .post(requestBody)
                .build();

        BulkDiagnoseResponse bulkDiagnoseResponse = null;
        // 发送请求并处理响应
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() && response.body() != null) {
                String responseBody = response.body().string();
               log.info("响应数据: {}", responseBody);
                // 处理返回的 JSON 数据
                bulkDiagnoseResponse = objectMapper.readValue(responseBody, BulkDiagnoseResponse.class);
                bulkDiagnoseResponse.setPredictionResultsSize(bulkDiagnoseResponse.getPredictionResults().size());
                log.info(bulkDiagnoseResponse.toString());
            } else {
                System.out.println("请求失败: " + response);
            }
        }
        return bulkDiagnoseResponse;
    }

    public PersonalDiagnoseResponse requestForPersonalDiagnose(List<File> files) throws IOException {
        String methodUrl = "/mock/file-personal";
        if (files == null || files.size() == 0) {
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
                    //MediaType.get("application/zip"), // 根据文件类型设置 MIME 类型
                    file
            );
            // 添加文件到表单字段 "fileList"
            requestBodyBuilder.addPart(
                    Headers.of("Content-Disposition", "form-data; name=\"images\"; filename=\"" + file.getName() + "\""),
                    fileBody
            );
        }

        // 构建请求体
        RequestBody requestBody = requestBodyBuilder.build();

        // 创建请求
        Request request = new Request.Builder()
                .url(baseUrl + methodUrl) // 替换为你的服务端地址
                .post(requestBody)
                .build();

        PersonalDiagnoseResponse personalDiagnoseResponse = null;
        // 发送请求并处理响应
        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful() && response.body() != null) {
                String responseBody = response.body().string();
                log.info("响应数据: {}", responseBody);
                // 处理返回的 JSON 数据
                personalDiagnoseResponse = objectMapper.readValue(responseBody, PersonalDiagnoseResponse.class);
                log.info(personalDiagnoseResponse.toString());
            } else {
                System.out.println("请求失败: " + response);
            }
        }
        return personalDiagnoseResponse;
    }
}
