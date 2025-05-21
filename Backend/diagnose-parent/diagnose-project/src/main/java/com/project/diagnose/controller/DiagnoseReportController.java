package com.project.diagnose.controller;

import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.dto.query.DiagnoseQuery;
import com.project.diagnose.dto.vo.*;
import com.project.diagnose.service.DiagnoseService;
import com.project.diagnose.utils.FileUtils;
import com.project.diagnose.utils.RedisUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/diagnose")
public class DiagnoseReportController {
    @Value("${aliyun.oss.bucketName}")
    private String bucket;

    @Autowired
    private DiagnoseService diagnoseService;
    @Autowired
    private RedisUtils redisUtils;

    @PostMapping("/upload/zip")
    //@PreAuthorize("hasAuthority('upload')")
    @LogAnnotation(module = "FileController",operator = "用MinIO批量上传诊断压缩包")
    public Result<List<DiagnoseImageVo>> uploadZip(@RequestHeader("Authorization") String token, @RequestParam("fileList") MultipartFile[] files) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        List<DiagnoseImageVo> diagnoseImageVos = diagnoseService.uploadFiles(bucket, files, FileUtils.Category.CATEGORY_ZIP, userId);
        return Result.success(diagnoseImageVos);
    }

    @PostMapping("/upload/image")
    //@PreAuthorize("hasAuthority('upload')")
    @LogAnnotation(module = "FileController",operator = "用MinIO批量上传诊断图片")
    public Result<List<DiagnoseImageVo>> uploadImage(@RequestHeader("Authorization") String token, @RequestParam("fileList") MultipartFile[] files) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        List<DiagnoseImageVo> diagnoseImageVos = diagnoseService.uploadFiles(bucket, files, FileUtils.Category.CATEGORY_IMAGE, userId);
        return Result.success(diagnoseImageVos);
    }

    @PostMapping("/create-bulk")
    @LogAnnotation(module = "FileController",operator = "生成诊断报告")
    public Result<DiagnoseResponseVo> createBulkReport(@RequestHeader("Authorization") String token, @RequestBody List<String> idList) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        DiagnoseResponseVo diagnose = diagnoseService.generateBulkDiagnoseReport(userId, idList);

        return Result.success(diagnose);
    }

    @PostMapping("/create-personal")
    @LogAnnotation(module = "FileController",operator = "生成诊断报告")
    public Result<DiagnoseResponseVo> createPersonalReport(@RequestHeader("Authorization") String token, @RequestBody List<String> idList) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        DiagnoseResponseVo diagnoseResponseVo = diagnoseService.generatePersonalDiagnoseReport(userId, idList);

        return Result.success(diagnoseResponseVo);
    }

    @GetMapping("/details")
    @LogAnnotation(module = "FileController",operator = "查询报告详情")
    public Result<DiagnoseResponseVo> BulkDetails(@RequestHeader("Authorization") String token, @RequestParam("diagnoseId") Long diagnoseId) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        DiagnoseResponseVo diagnoseResponseVo = diagnoseService.getDiagnoseDetails(userId, diagnoseId);
        return Result.success(diagnoseResponseVo);
    }

    @PostMapping("/history")
    //@PreAuthorize("hasAuthority('upload')")
    @LogAnnotation(module = "FileController",operator = "查询历史诊断报告")
    public Result<PageVo<DiagnoseBaseVo>> upload(@RequestHeader("Authorization") String token, @RequestBody DiagnoseQuery diagnoseQuery) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        PageVo<DiagnoseBaseVo> diagnoseReportVoPage = diagnoseService.getDiagnoseHistory(userId, diagnoseQuery);
        return Result.success(diagnoseReportVoPage);
    }



}