package com.project.diagnose.dto.vo;

import com.project.diagnose.dto.response.DiagnoseResponse;
import lombok.Data;

import java.util.List;

@Data
public class DiagnoseReportVo {
    private String id;
    private String username;
    private String userId;
    private String time;
    private DiagnoseResponse report;
    private List<String> urlList;


}
