package com.project.diagnose.utils;

import com.project.diagnose.dto.response.UploadFileResponse;
import com.project.diagnose.pojo.DiagnoseFile;
import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import java.io.InputStream;
import java.util.HashMap;

@Component
public class UploadFileUtilsFactory {
    @Autowired
    private ApplicationContext context;

    private HashMap<StorageSource, UploadFileUtils> uploadFileUtilsMap;

    @AllArgsConstructor
    @Getter
    public enum StorageSource {
        MINIO("minio"),
        ALI_OSS("ali_oss");

        private final String value;

        public static StorageSource getStorageSource(String source) {
            for (StorageSource storageSource : StorageSource.values()) {
                if (storageSource.getValue().equals(source)) {
                    return storageSource;
                }
            }
            return null;
        }
    }

    // 初始化hashmap
    @PostConstruct
    public void init() {
        uploadFileUtilsMap = new HashMap<>();
        // 添加更多的类型映射
        uploadFileUtilsMap.put(StorageSource.MINIO, context.getBean(MinioUtils.class));
        uploadFileUtilsMap.put(StorageSource.ALI_OSS, context.getBean(AliOSSUtils.class));
    }

    public UploadFileResponse upload(MultipartFile file, StorageSource storageSource, String bucket) throws Exception {
        // 根据storageSource动态获取上传文件的工具类
        UploadFileUtils uploadFileUtils = uploadFileUtilsMap.get(storageSource);
        // 上传文件
        UploadFileResponse response = uploadFileUtils.upload(file, bucket);
        // 添加文件的存储源信息
        response.setStorageSource(storageSource.getValue());
        return response;
    }
    public UploadFileResponse upload(InputStream file, StorageSource storageSource, String bucket, String name, String mimeType) throws Exception {
        // 根据storageSource动态获取上传文件的工具类
        UploadFileUtils uploadFileUtils = uploadFileUtilsMap.get(storageSource);
        // 上传文件
        UploadFileResponse response = uploadFileUtils.upload(file, bucket, name, mimeType);
        // 添加文件的存储源信息
        response.setStorageSource(storageSource.getValue());
        return response;
    }

    public void delete(DiagnoseFile diagnoseFile) throws Exception {
        StorageSource storageSource = StorageSource.getStorageSource(diagnoseFile.getStorageSource());
        String bucket = diagnoseFile.getBucket();
        String objectPath = diagnoseFile.getObjectPath();

        // 根据storageSource动态获取删除文件的工具类
        UploadFileUtils uploadFileUtils = uploadFileUtilsMap.get(storageSource);
        // 删除文件
        uploadFileUtils.delete(bucket, objectPath);

    }

    public InputStream download(DiagnoseFile diagnoseFile) throws Exception {
        StorageSource storageSource = StorageSource.getStorageSource(diagnoseFile.getStorageSource());
        String bucket = diagnoseFile.getBucket();
        String objectPath = diagnoseFile.getObjectPath();

        UploadFileUtils uploadFileUtils = uploadFileUtilsMap.get(storageSource);
        return uploadFileUtils.download(bucket, objectPath);
    }
}
