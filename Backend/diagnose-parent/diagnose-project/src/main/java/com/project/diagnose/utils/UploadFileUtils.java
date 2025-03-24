package com.project.diagnose.utils;

import com.project.diagnose.dto.response.UploadFileResponse;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;

public interface UploadFileUtils {
    UploadFileResponse upload(MultipartFile file, String bucket);
    UploadFileResponse upload(InputStream inputStream, String bucket, String originalFilename, String mimeType);
    void delete(String bucket, String objectName);
    InputStream download(String bucket, String objectName);
}
