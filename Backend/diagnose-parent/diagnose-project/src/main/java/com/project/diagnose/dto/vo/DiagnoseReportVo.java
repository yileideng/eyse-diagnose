package com.project.diagnose.dto.vo;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class DiagnoseReportVo {
    private String id;

    private String userId;

    private String time;

    private String reportResultId;

    private String diagnoseMode;
}
