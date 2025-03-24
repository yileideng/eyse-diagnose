package com.project.diagnose.service.Impl;

import cn.hutool.core.lang.Assert;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.diagnose.dto.query.AvatarQuery;
import com.project.diagnose.dto.response.UploadFileResponse;
import com.project.diagnose.dto.vo.PageVo;
import com.project.diagnose.dto.vo.AvatarImageVo;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.mapper.AvatarImageMapper;
import com.project.diagnose.pojo.AvatarImage;
import com.project.diagnose.service.AvatarImageService;
import com.project.diagnose.utils.AliOSSUtils;
import com.project.diagnose.utils.FileUtils;
import com.project.diagnose.utils.MinioUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;


/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author itcast
 */
@Slf4j
@Service
public class AvatarImageServiceImpl extends ServiceImpl<AvatarImageMapper, AvatarImage> implements AvatarImageService {
    @Autowired
    private MinioUtils minioUtils;
    @Autowired
    private AliOSSUtils aliOSSUtils;
    @Autowired
    private AvatarImageMapper avatarImageMapper;



    @Override
    public String uploadAndInsert(String bucket, MultipartFile file, FileUtils.Category requiredCategory, Long userId) {
        if (file.isEmpty()) {
            throw new DiagnoseException("上传的文件为空");
        }
        String fileName = file.getOriginalFilename();
        if(!FileUtils.checkFileCategory(fileName, requiredCategory)){
            throw new DiagnoseException("上传的文件类型不符合要求");
        }

        // 上传文件到Minio
        UploadFileResponse response = null;
        try {
            response = minioUtils.upload(file, bucket);
            if(response!=null){
                log.info("上传文件成功");
            }
        } catch (Exception e) {
            throw new DiagnoseException("上传文件到Minio失败, fileName: " + fileName);
        }

// 向数据库中插入上传文件的信息

        AvatarImage avatarImage = new AvatarImage();

        // 设置文件分类:音频,图片等等
        avatarImage.setCategory(requiredCategory.getCategory());
        // 设置上传文件的用户
        avatarImage.setUserId(userId);
        // 设置文件创建时间
        avatarImage.setTime(LocalDateTime.now());
        avatarImage.setStorageSource(response.getStorageSource());
        avatarImage.setBucket(bucket);
        avatarImage.setObjectPath(response.getObjectPath());
        // 设置文件的访问路径
        avatarImage.setUrl(response.getUrl());
        // 设置文件名
        avatarImage.setName(fileName);


        // 写入UploadFile表
        avatarImageMapper.insert(avatarImage);
        log.info("向数据库插入文件成功");

        return response.getUrl();
    }

    @Override
    public PageVo<AvatarImageVo> getPageByCategory(AvatarQuery avatarQuery, Long userId) {
        // Assert.notNull 方法会抛出一个 IllegalArgumentException 异常
        Assert.notNull(avatarQuery, "用户参数不能为空");

        String fileName = avatarQuery.getName();

        //默认按照time降序排序(如果有参数query,就按照参数排序)
        Page<AvatarImage> page= avatarQuery.toMpPage();

        Page<AvatarImage> p = lambdaQuery()
                .eq(AvatarImage::getUserId, userId)
                .like(fileName!=null, AvatarImage::getName, fileName)
                .page(page);


        //封装成Vo:Po<Po>转为Vo<Vo>
        return PageVo.of(p, AvatarImageVo.class);
    }
}

