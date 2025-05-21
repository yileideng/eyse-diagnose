package com.project.diagnose.controller;


import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.dto.vo.Result;
import com.project.diagnose.service.AvatarImageService;
import com.project.diagnose.utils.FileUtils;
import com.project.diagnose.utils.RedisUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequestMapping("/upload")
public class AvatarImageController {
    @Value("${aliyun.oss.bucketName}")
    private String bucket;

    @Autowired
    private AvatarImageService avatarImageService;
    @Autowired
    private RedisUtils redisUtils;


    @PostMapping("/avatar")
    //@PreAuthorize("hasAuthority('upload')")
    @LogAnnotation(module = "FileController",operator = "用MinIO上传用户头像")
    public Result<String> upload(@RequestHeader("Authorization") String token, @RequestParam("file") MultipartFile image) {
        Long userId = redisUtils.getLoginUserInRedis(token).getUser().getId();
        String url = avatarImageService.uploadAndInsert(bucket, image, FileUtils.Category.CATEGORY_IMAGE, userId);
        return Result.success(url);
    }
}
