package com.project.diagnose.dto.vo;

import com.project.diagnose.dto.response.DiagnoseResponseList;
import lombok.Data;

import java.util.List;

@Data
public class DiagnoseResponseVo {
    // 报告基本信息
    private String id;
    private String time;
    //用户信息
    private String userId;
    private String username;
    private String email;
    private String phoneNumber;
    private String avatarUrl;

    // 报告结果
    private DiagnoseResponseList report;
    private List<String> urlList;


}
