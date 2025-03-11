package com.project.diagnose.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.project.diagnose.dto.query.DiagnoseQuery;
import com.project.diagnose.dto.vo.DiagnoseImageVo;
import com.project.diagnose.dto.vo.DiagnoseReportVo;
import com.project.diagnose.dto.vo.PageVo;
import com.project.diagnose.pojo.DiagnoseImage;
import com.project.diagnose.utils.FileUtils;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

public interface DiagnoseService extends IService<DiagnoseImage> {
    List<DiagnoseImageVo> uploadImages(String bucket, MultipartFile[] files, FileUtils.Category requiredCategory, Long userId);

    DiagnoseReportVo generateDiagnoseReport(Long userId, List<String> idList);

    PageVo<DiagnoseReportVo> getDiagnoseHistory(Long userId, DiagnoseQuery diagnoseQuery);

    DiagnoseReportVo getDiagnoseDetails(Long userId, Long diagnoseId);
}
