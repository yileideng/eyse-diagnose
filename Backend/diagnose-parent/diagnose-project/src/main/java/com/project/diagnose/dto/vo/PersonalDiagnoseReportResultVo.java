package com.project.diagnose.dto.vo;

import com.project.diagnose.dto.response.BulkDiagnoseResponse;
import com.project.diagnose.dto.response.PersonalDiagnoseResponse;
import lombok.Data;

import java.util.List;

@Data
public class PersonalDiagnoseReportResultVo {
    private String id;
    private String username;
    private String userId;
    private String time;
    private PersonalDiagnoseResponse report;
    private List<String> urlList;
}
