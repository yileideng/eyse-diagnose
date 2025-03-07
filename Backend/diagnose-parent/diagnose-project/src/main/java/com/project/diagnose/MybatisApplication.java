package com.project.diagnose;

import com.aliyun.oss.OSS;
import com.aliyun.oss.OSSClientBuilder;
import com.project.diagnose.utils.AliOSSProperties;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.ServletComponentScan;
import org.springframework.context.annotation.Bean;

@ServletComponentScan
@SpringBootApplication
public class MybatisApplication {

    public static void main(String[] args) {
        SpringApplication.run(MybatisApplication.class, args);
    }

    @Autowired
    private AliOSSProperties aliOSSProperties;

    @Bean
    public OSS ossClient() {
        return new OSSClientBuilder().build(
                aliOSSProperties.getEndpoint(),
                aliOSSProperties.getAccessKeyId(),
                aliOSSProperties.getAccessKeySecret()
        );
    }
}
