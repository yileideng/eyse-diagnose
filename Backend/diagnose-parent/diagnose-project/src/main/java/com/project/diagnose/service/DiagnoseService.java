package com.project.diagnose.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.project.diagnose.dto.query.DiagnoseQuery;
import com.project.diagnose.dto.vo.*;
import com.project.diagnose.pojo.DiagnoseImage;
import com.project.diagnose.utils.FileUtils;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

public interface DiagnoseService extends IService<DiagnoseImage> {
    List<DiagnoseImageVo> uploadFiles(String bucket, MultipartFile[] files, FileUtils.Category requiredCategory, Long userId);

    PageVo<DiagnoseReportVo> getDiagnoseHistory(Long userId, DiagnoseQuery diagnoseQuery);

    BulkDiagnoseReportResultVo generateBulkDiagnoseReport(Long userId, List<String> idList);
    BulkDiagnoseReportResultVo getBulkDiagnoseDetails(Long userId, Long diagnoseId);

    BulkDiagnoseReportResultVo generatePersonalDiagnoseReport(Long userId, List<String> idList);
    BulkDiagnoseReportResultVo getPersonalDiagnoseDetails(Long userId, Long diagnoseId);
}
