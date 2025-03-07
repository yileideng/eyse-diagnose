/*
package com.project.diagnose;

import com.aliyun.oss.OSS;
import com.aliyun.oss.model.OSSObject;
import com.aliyun.oss.model.OSSObjectSummary;
import com.aliyun.oss.model.ObjectListing;
import com.project.diagnose.pojo.User;
import com.project.diagnose.service.UserService;
import com.project.diagnose.utils.AliOSSProperties;
import com.project.diagnose.utils.MinioUtils;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.errors.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.io.*;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.UUID;


@SpringBootTest
public class Test {
    @Autowired
    private UserService userService;
    @Autowired
    private OSS ossClient;
    @Autowired
    private AliOSSProperties aliOSSProperties;


    // 测试主键回显
    @org.junit.jupiter.api.Test
    void testMainKeyReturn(){
        User user=new User();

        user.setUsername("MainKey2");
        user.setPassword("1234");
        // userMapper.insert(user);
        userService.save(user);
        if(user.getId()!=null){
            System.out.println("userId:"+user.getId());
        }else {
            System.out.println("主键回显失败");
        }
    }

    @org.junit.jupiter.api.Test
    void getOSSFiles(){
        ObjectListing objectListing = ossClient.listObjects(aliOSSProperties.getBucketName());
        for (OSSObjectSummary objectSummary : objectListing.getObjectSummaries()) {
            System.out.println(" - " + objectSummary.getKey() + " (大小 = " + objectSummary.getSize() + ")");
        }
    }
    //c215eb9c-5789-4b82-b7dc-245e6b195b94.pptx
    //aac2268b-fed8-44f8-bffb-3ef1c5b42f52.pptx
    //147ce3e8-872a-4c3b-8892-af066e8519e7.pptx

    @org.junit.jupiter.api.Test
    public void download(){
        String url = "https://dyl123.oss-cn-chengdu.aliyuncs.com/147ce3e8-872a-4c3b-8892-af066e8519e7.pptx";
        String fileName = null;
        try {
            // 获取阿里云OSS参数
            String bucketName=aliOSSProperties.getBucketName();
            // 从 URL 中提取文件名
            fileName = extractFilenameFromUrl(url);

            OSSObject ossObject = ossClient.getObject(bucketName, fileName);

        } catch (Exception e) {
            System.out.println("文件名:"+fileName);
            throw new RuntimeException("Failed to download file from OSS: " + fileName, e);
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
            return url; // 如果没有斜杠，整个URL就是文件名
        }
    }



}


*/
