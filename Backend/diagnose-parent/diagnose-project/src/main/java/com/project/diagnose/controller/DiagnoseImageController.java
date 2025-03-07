package com.project.diagnose.controller;

import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.dto.vo.Result;
import com.project.diagnose.dto.vo.UploadFileVo;
import com.project.diagnose.pojo.UploadFile;
import com.project.diagnose.service.AvatarImageService;
import com.project.diagnose.utils.RedisUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequestMapping("/diagnose/upload")
public class DiagnoseImageController {
    @Value("${minio.bucket.image}")
    private String bucket;

    @Autowired
    private AvatarImageService avatarImageService;
    @Autowired
    private RedisUtils redisUtils;


    @PostMapping("/avatar")
    //@PreAuthorize("hasAuthority('upload')")
    @LogAnnotation(module = "FileController",operator = "用MinIO上传用户头像")
    public Result<UploadFileVo> upload(@RequestHeader("Authorization") String token, @RequestParam("file") MultipartFile image) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        UploadFileVo avatarImage = avatarImageService.uploadAndInsert(bucket, image, UploadFile.Category.CATEGORY_AVATAR, userId, null);
        return Result.success(avatarImage);
    }
}