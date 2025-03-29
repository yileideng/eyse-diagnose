package com.project.diagnose.dto.vo;

import com.project.diagnose.dto.response.BulkDiagnoseResponse;
import lombok.Data;

import java.util.List;

@Data
public class BulkDiagnoseReportResultVo {
    private String id;
    private String username;
    private String userId;
    private String time;
    private BulkDiagnoseResponse report;
    private List<String> urlList;


}
