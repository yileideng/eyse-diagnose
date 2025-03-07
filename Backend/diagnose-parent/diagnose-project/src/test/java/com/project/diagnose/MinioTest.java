package com.project.diagnose;

import com.project.diagnose.utils.FileUtils;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.errors.*;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.UUID;

public class MinioTest {

    private MinioClient minioClient = MinioClient.builder()
            .endpoint("http://8.137.104.3:8000")
            .credentials("admin", "admin123")
            .build();

    @Test
    void minioUpload() throws IOException, ServerException, InsufficientDataException, ErrorResponseException, NoSuchAlgorithmException, InvalidKeyException, InvalidResponseException, XmlParserException, InternalException {
        // 获取文件原始名称
        String originalFilename = "C:/Users/18101/Pictures/Saved Pictures/IMG_20231009_164259.png";
        System.out.println(originalFilename);
        File file =new File(originalFilename);
        // 获取文件的输入流
        InputStream inputStream = new FileInputStream(file);


        // 获取文件mimeType
        String mimeType= FileUtils.getMimeType(originalFilename);
        System.out.println(mimeType);

        // 根据当前时间生成文件目录
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        String datePath = sdf.format(new Date()).replace("-", "/");
        System.out.println(datePath);
        // 生成唯一的文件路径
        String filePath = mimeType+ "/" + datePath + "/" + UUID.randomUUID() + originalFilename;
        System.out.println(filePath);

        // 上传文件
        minioClient.putObject(PutObjectArgs.builder()
                .bucket("upload") // 存储桶名称
                .object(filePath) // 对象名称（文件名）
                .stream(inputStream, inputStream.available(), -1) // 输入流、文件大小、分块大小（-1 表示自动）
                .contentType(mimeType) // 设置 Content-Type
                .build()
        );

        // 创建文件访问路径
        String url = "http://8.137.104.3:8001" + "/" + "upload" + "/" + filePath;
        System.out.println(url);

    }
}
