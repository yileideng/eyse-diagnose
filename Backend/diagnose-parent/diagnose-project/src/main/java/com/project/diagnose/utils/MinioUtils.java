package com.project.diagnose.utils;

import io.minio.*;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.UUID;

@Slf4j
@Configuration
public class MinioUtils {

    @Value("${minio.endpoint}")
    private String endpoint;

    @Autowired
    private MinioClient minioClient;


    public String uploadAndGetUrl(MultipartFile file, String bucket) throws Exception {
        log.info(bucket);
        // 获取文件的输入流
        InputStream inputStream = file.getInputStream();
        // 获取文件原始名称
        String originalFilename = file.getOriginalFilename();
        // 获取文件mimeType
        String mimeType=FileUtils.getMimeType(originalFilename);

        log.info(mimeType);

        // 根据当前时间生成文件目录
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        String datePath = sdf.format(new Date()).replace("-", "/");
        // 生成唯一的文件路径
        String filePath = mimeType+ "/" + datePath + "/" + UUID.randomUUID() + originalFilename;
        log.info(filePath);

        // 上传文件
        minioClient.putObject(PutObjectArgs.builder()
                .bucket(bucket) // 存储桶名称
                .object(filePath) // 对象名称（文件名）
                .stream(inputStream, file.getSize(), -1) // 输入流、文件大小、分块大小（-1 表示自动）
                .contentType(mimeType) // 设置 Content-Type
                .build()
        );

        String url = endpoint + "/" + bucket + "/" + filePath;

        return url;
    }


/*    public String uploadFileToMinio(String filePath, String bucket) throws Exception {
        String url = minioClient.getPresignedObjectUrl(
                GetPresignedObjectUrlArgs.builder()
                        .method(Method.GET)
                        .bucket(bucket)
                        .object(filePath)
                        .expiry(7 * 24 * 60 * 60).build() // url过期时间为最大七天
        );
        return url;
    }*/

/*    public String uploadAndGetUrl(String bucket, MultipartFile file) throws Exception {
        String filePath = uploadFileToMinio(file, bucket);
        String url = uploadFileToMinio(filePath, bucket);
        return url;
    }*/

}
