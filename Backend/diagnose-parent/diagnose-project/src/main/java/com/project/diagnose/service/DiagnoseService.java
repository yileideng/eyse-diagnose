package com.project.diagnose.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.project.diagnose.dto.query.DiagnoseQuery;
import com.project.diagnose.dto.vo.*;
import com.project.diagnose.pojo.DiagnoseFile;
import com.project.diagnose.utils.FileUtils;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

public interface DiagnoseService extends IService<DiagnoseFile> {
    List<DiagnoseImageVo> uploadFiles(String bucket, MultipartFile[] files, FileUtils.Category requiredCategory, Long userId);

    PageVo<DiagnoseBaseVo> getDiagnoseHistory(Long userId, DiagnoseQuery diagnoseQuery);

    DiagnoseResponseVo generateBulkDiagnoseReport(Long userId, List<String> idList);
    DiagnoseResponseVo getDiagnoseDetails(Long userId, Long diagnoseId);

    DiagnoseResponseVo generatePersonalDiagnoseReport(Long userId, List<String> idList);
}
