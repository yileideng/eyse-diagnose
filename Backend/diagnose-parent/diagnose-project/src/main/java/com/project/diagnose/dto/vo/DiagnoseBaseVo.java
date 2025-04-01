package com.project.diagnose.dto.vo;

import lombok.Data;

@Data
public class DiagnoseBaseVo {
    private String id;

    private String userId;

    private String time;

    private String reportResultId;

    private String diagnoseMode;
}
