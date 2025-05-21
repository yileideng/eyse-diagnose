package com.project.diagnose.utils;

import com.aliyun.oss.OSS;
import com.aliyun.oss.model.OSSObject;
import com.project.diagnose.dto.response.UploadFileResponse;
import com.project.diagnose.exception.DiagnoseException;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.xslf.usermodel.XMLSlideShow;
import org.apache.poi.xslf.usermodel.XSLFShape;
import org.apache.poi.xslf.usermodel.XSLFTextShape;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * 阿里云 OSS 工具类
 */
@Slf4j
@Data
@Component
@Configuration()
public class AliOSSUtils implements UploadFileUtils{
    @Value("${aliyun.oss.endpoint}")
    private String endpoint;
    @Value("${aliyun.oss.bucketName}")
    private String bucketName;

    @Autowired
    private OSS ossClient;

    public String getBucket() {
        return bucketName;
    }

    // 下载文件
    public OSSObject download(String url){
        try {
            // 从 URL 中提取文件名
            String fileName = extractFilenameFromUrl(url);

            OSSObject ossObject = ossClient.getObject(bucketName, fileName);

            return ossObject;
        } catch (Exception e) {
            throw new RuntimeException("Failed to download file from OSS, fileUrl: " + url, e);
        }
    }
    // 提取文件名的方法
    private String extractFilenameFromUrl(String url) {
        url = url.trim().replaceAll("\"", "");

        // 使用lastIndexOf方法找到最后一个斜杠的位置
        int lastSlashIndex = url.lastIndexOf('/');

        // 如果找到斜杠，提取斜杠之后的部分；否则返回整个URL
        if (lastSlashIndex != -1) {
            return url.substring(lastSlashIndex + 1);
        } else {
            // 如果没有斜杠，整个URL就是文件名
            return url;
        }
    }


    // 上传文件
    @Override
    public UploadFileResponse upload(MultipartFile file, String bucketName) {
        if(!checkBucketName(bucketName)){
            throw new RuntimeException("bucket:" + bucketName + "不存在");
        }
        // 获取上传的文件的输入流
        try(InputStream inputStream = file.getInputStream()) {
            // 避免文件覆盖
            String originalFilename = file.getOriginalFilename();
            String fileName = UUID.randomUUID().toString() + originalFilename.substring(originalFilename.lastIndexOf("."));
            log.info("文件名称: {}", fileName);

            ossClient.putObject(bucketName, fileName, inputStream);

            //文件访问路径
            String url = endpoint.split("//")[0] + "//" + bucketName + "." + endpoint.split("//")[1] + "/" + fileName;

            UploadFileResponse uploadFileResponse = new UploadFileResponse();
            uploadFileResponse.setUrl(url);
            uploadFileResponse.setBucket(bucketName);
            uploadFileResponse.setObjectPath(fileName);
            uploadFileResponse.setName(file.getOriginalFilename());
            uploadFileResponse.setStorageSource("ali_oss");

            // 把上传到oss的路径返回
            return uploadFileResponse;
        }catch (Exception e){
            throw new DiagnoseException("上传文件失败,error:" + e.getMessage());
        }

    }

    private Boolean checkBucketName(String bucket) {
        return bucket.equals(bucketName);
    }

    @Override
    public void delete(String bucketName, String objectName) {
        ossClient.deleteObject(bucketName, objectName);
    }

    @Override
    public InputStream download(String bucket, String objectName){
        OSSObject object = ossClient.getObject(bucketName, objectName);
        return object.getObjectContent();
    }

    @Override
    public UploadFileResponse upload(InputStream inputStream, String bucket, String originalFilename, String mimeType) {
        return null;
    }

}
