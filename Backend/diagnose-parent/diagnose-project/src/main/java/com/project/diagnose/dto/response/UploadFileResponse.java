package com.project.diagnose.dto.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class UploadFileResponse {
    private String name;

    private String storageSource;

    private String bucket;

    private String objectPath;

    private String url;
}
