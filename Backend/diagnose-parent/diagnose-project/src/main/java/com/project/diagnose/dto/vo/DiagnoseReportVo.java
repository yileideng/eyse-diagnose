package com.project.diagnose.dto.vo;

import lombok.Data;

import java.util.List;

@Data
public class DiagnoseReportVo {
    private String id;
    private String username;
    private String userId;
    private String time;
    private String report;
    private List<String> urlList;


}
