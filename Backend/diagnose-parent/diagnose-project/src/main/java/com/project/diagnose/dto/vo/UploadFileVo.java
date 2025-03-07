package com.project.diagnose.dto.vo;

import com.project.diagnose.pojo.UploadFile;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@NoArgsConstructor
@Data
public class UploadFileVo {
    /**
     * 文件id
     */
    private String id;

    /**
     * 文件名称
     */
    private String name;

    /**
     * 文件访问路径
     */
    private String url;

    /**
     * 上传文件的用户id
     */
    private String userId;

    private LocalDateTime time;

    private String category;

    public UploadFileVo(UploadFile uploadFile) {
        this.setId(uploadFile.getId().toString());
        this.setName(uploadFile.getName());
        this.setUrl(uploadFile.getUrl());
        this.setUserId(uploadFile.getUserId().toString());
        this.setTime(uploadFile.getTime());
        this.setCategory(uploadFile.getCategory());
    }
}
