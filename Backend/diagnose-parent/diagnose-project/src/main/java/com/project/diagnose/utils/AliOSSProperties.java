package com.project.diagnose.utils;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Data
@Component
@ConfigurationProperties(prefix = "aliyun.oss")//自动从yml中注入名字相同的属性
public class AliOSSProperties {
    private String endpoint;
    private String accessKeyId;
    private String accessKeySecret;
    private String bucketName;

    /*
    使用环境变量注入accessKeyId:
    public AliOSSProperties(){
        accessKeyId = System.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID");
    }
    */
}
