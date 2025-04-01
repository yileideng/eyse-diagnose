package com.project.diagnose.dto.vo;

import com.project.diagnose.pojo.DiagnoseFile;
import lombok.Data;
import lombok.NoArgsConstructor;

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



    public DiagnoseImageVo(DiagnoseFile diagnoseFile) {
        this.setId(diagnoseFile.getId().toString());
        this.setName(diagnoseFile.getName());
        this.setUrl(diagnoseFile.getUrl());
    }
}
