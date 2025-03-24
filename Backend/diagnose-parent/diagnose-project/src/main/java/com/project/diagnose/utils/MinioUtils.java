package com.project.diagnose.utils;

import com.project.diagnose.dto.response.UploadFileResponse;
import com.project.diagnose.exception.DiagnoseException;
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
public class MinioUtils  implements UploadFileUtils{

    @Value("${minio.endpoint}")
    private String endpoint;
    @Value("${minio.bucket.avatar}")
    private String diagnoseAvatar;
    @Value("${minio.bucket.diagnose}")
    private String diagnose;

    @Autowired
    private MinioClient minioClient;


    @Override
    public UploadFileResponse upload(MultipartFile file, String bucket) {
        log.info("bucket: {}", bucket);
        if(!checkBucket(bucket)) {
            throw new RuntimeException("bucket:" + bucket + "不存在");
        }

        // 获取文件的输入流
        try(InputStream inputStream = file.getInputStream()) {
            // 获取文件原始名称
            String originalFilename = file.getOriginalFilename();
            // 获取文件mimeType
            String mimeType = FileUtils.getMimeType(originalFilename);

            log.info("mimeType: {}", mimeType);

            // 根据当前时间生成文件目录
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
            String datePath = sdf.format(new Date()).replace("-", "/");
            // 生成唯一的文件路径
            String filePath = mimeType + "/" + datePath + "/" + UUID.randomUUID() + originalFilename;
            log.info("filePath: {}", filePath);

            // 上传文件
            minioClient.putObject(PutObjectArgs.builder()
                    .bucket(bucket) // 存储桶名称
                    .object(filePath) // 对象名称（文件名）
                    .stream(inputStream, file.getSize(), -1) // 输入流、文件大小、分块大小（-1 表示自动）
                    .contentType(mimeType) // 设置 Content-Type
                    .build()
            );
            log.info("上传文件到minio成功");

            String url = endpoint + "/" + bucket + "/" + filePath;

            UploadFileResponse uploadFileResponse = new UploadFileResponse();
            uploadFileResponse.setUrl(url);
            uploadFileResponse.setStorageSource("minio");
            uploadFileResponse.setBucket(bucket);
            uploadFileResponse.setObjectPath(filePath);
            uploadFileResponse.setName(file.getOriginalFilename());

            return uploadFileResponse;
        }catch (Exception e) {
            throw new DiagnoseException("上传文件失败:" + e.getMessage());
        }
    }
    @Override
    public UploadFileResponse upload(InputStream inputStream, String bucket, String originalFilename, String mimeType) {
        log.info("bucket: {}", bucket);
        if(!checkBucket(bucket)) {
            throw new RuntimeException("bucket:" + bucket + "不存在");
        }

        try {
            // 根据当前时间生成文件目录
            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
            String datePath = sdf.format(new Date()).replace("-", "/");
            // 生成唯一的文件路径
            String filePath = mimeType + "/" + datePath + "/" + UUID.randomUUID() + originalFilename;
            log.info("filePath: {}", filePath);
            // 上传文件
            minioClient.putObject(PutObjectArgs.builder()
                    .bucket(bucket) // 存储桶名称
                    .object(filePath) // 对象名称（文件名）
                    .stream(inputStream, inputStream.available(), -1) // 输入流、文件大小、分块大小（-1 表示自动）
                    .contentType(mimeType) // 设置 Content-Type
                    .build()
            );
            log.info("上传文件到minio成功");

            String url = endpoint + "/" + bucket + "/" + filePath;

            UploadFileResponse uploadFileResponse = new UploadFileResponse();
            uploadFileResponse.setUrl(url);
            uploadFileResponse.setBucket(bucket);
            uploadFileResponse.setObjectPath(filePath);
            uploadFileResponse.setName(originalFilename);

            return uploadFileResponse;
        }catch (Exception e) {
            throw new DiagnoseException("上传文件失败:" + e.getMessage());
        }
    }

    private Boolean checkBucket(String bucket) {
        return bucket.equals(diagnose) || bucket.equals(diagnoseAvatar);
    }

    @Override
    public void delete(String bucket, String objectName) {
        try {
            minioClient.removeObject(
                    RemoveObjectArgs.builder()
                            .bucket(bucket)
                            .object(objectName)
                            .build());
        }catch (Exception e){
            throw new DiagnoseException("删除文件失败, bucket:" + bucket + ",objectName:" + objectName + "errorMessage:" + e.getMessage());
        }

    }

    @Override
    public InputStream download(String bucket, String objectName) {
        // 从MinIO下载文件
        try {
            return minioClient.getObject(
                    GetObjectArgs.builder()
                            .bucket(bucket)
                            .object(objectName)
                            .build());
        } catch (Exception e) {
            throw new DiagnoseException("从minio下载文件失败");
        }
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
