package com.project.diagnose.utils;

import com.aliyun.oss.OSS;
import com.aliyun.oss.model.OSSObject;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.xslf.usermodel.XMLSlideShow;
import org.apache.poi.xslf.usermodel.XSLFShape;
import org.apache.poi.xslf.usermodel.XSLFTextShape;
import org.springframework.beans.factory.annotation.Autowired;
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
public class AliOSSUtils {

/*
@value注解只能一个一个的进行外部属性注入
而@ConfigurationProperties注解可以批量的将外部属性注入到bean对象的属性中
    @Value("${aliyun.oss.endpoint}")
    private String endpoint;
    @Value("${aliyun.oss.accessKeyId}")
    private String accessKeyId;
    @Value("${aliyun.oss.accessKeySecret}")
    private String accessKeySecret;
    @Value("${aliyun.oss.bucketName}")
    private String bucketName;
*/
    @Autowired
    private AliOSSProperties aliOSSProperties;

    @Autowired
    private OSS ossClient;

    // 下载文件
    public OSSObject download(String url){
        try {
            // 获取阿里云OSS参数
            String bucketName=aliOSSProperties.getBucketName();
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

    // 解析pptx文件
    public List<List<String>> extractTextFromPPT(OSSObject ossObject) throws Exception {
        try (InputStream inputStream = ossObject.getObjectContent()) {
            XMLSlideShow ppt = new XMLSlideShow(inputStream);
            // 外层列表存储每一页的文本列表
            List<List<String>> pptTextList = new ArrayList<>();

            // 遍历每一页幻灯片
            for (int i = 0; i < ppt.getSlides().size(); i++) {
                // 内层列表存储当前页的文本内容
                List<String> pageTextList = new ArrayList<>();

                // 遍历当前页的形状
                for (XSLFShape shape : ppt.getSlides().get(i)) {
                    if (shape instanceof XSLFTextShape) {
                        XSLFTextShape textShape = (XSLFTextShape) shape;
                        String text = textShape.getText();
                        if (text != null && !text.trim().isEmpty()) {
                            // 添加当前形状的文本到当前页的列表
                            pageTextList.add(text.trim());
                        }
                    }
                }

                // 将当前页的文本列表添加到外层列表
                pptTextList.add(pageTextList);
            }

            return pptTextList;
        } catch (Exception e) {
            log.info("Failed to extract text from PPT", e);
            throw new RuntimeException("Failed to extract text from PPT", e);
        }
    }

    // 上传文件
    public String upload(MultipartFile file) throws IOException {
        // 获取阿里云OSS参数
        String endpoint=aliOSSProperties.getEndpoint();
        String bucketName=aliOSSProperties.getBucketName();

        // 获取上传的文件的输入流
        InputStream inputStream = file.getInputStream();

        // 避免文件覆盖
        String originalFilename = file.getOriginalFilename();
        String fileName = UUID.randomUUID().toString() + originalFilename.substring(originalFilename.lastIndexOf("."));
        log.info("文件名称: {}", fileName);

        ossClient.putObject(bucketName, fileName, inputStream);

        //文件访问路径
        String url = endpoint.split("//")[0] + "//" + bucketName + "." + endpoint.split("//")[1] + "/" + fileName;

        // 把上传到oss的路径返回
        return url;
    }

}
