package com.project.diagnose.dto.vo;

import com.project.diagnose.pojo.AvatarImage;
import com.project.diagnose.pojo.DiagnoseImage;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
public class DiagnoseImageVo {
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



    public DiagnoseImageVo(DiagnoseImage diagnoseImage) {
        this.setId(diagnoseImage.getId().toString());
        this.setName(diagnoseImage.getName());
        this.setUrl(diagnoseImage.getUrl());
    }
}
