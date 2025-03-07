package com.project.diagnose.service.Impl;

import cn.hutool.core.lang.Assert;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.diagnose.dto.query.UploadFileQuery;
import com.project.diagnose.dto.vo.PageVo;
import com.project.diagnose.dto.vo.UploadFileVo;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.mapper.UploadFileMapper;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.UploadFile;
import com.project.diagnose.service.UploadFileService;
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
public class UploadFileServiceImpl extends ServiceImpl<UploadFileMapper, UploadFile> implements UploadFileService {
    @Autowired
    private MinioUtils minioUtils;
    @Autowired
    private AliOSSUtils aliOSSUtils;
    @Autowired
    private UploadFileMapper uploadFileMapper;
    @Autowired
    private UserMapper userMapper;


    @Override
    public UploadFileVo uploadAndInsert(String bucket, MultipartFile file, UploadFile.Category requiredCategory, Long userId, String voiceSeedName) {
        if (file.isEmpty()) {
            throw new DiagnoseException("上传的文件为空");
        }
        String fileName = file.getOriginalFilename();
        if(!FileUtils.checkFileCategory(fileName, requiredCategory)){
            throw new DiagnoseException("上传的文件类型不符合要求");
        }

        // 上传文件到Minio
        String url = null;
        try {
            url = minioUtils.uploadAndGetUrl(file, bucket);
            if(url!=null){
                log.info("上传文件成功");
            }
        } catch (Exception e) {
            throw new DiagnoseException("上传文件到Minio失败, fileName: " + fileName);
        }

// 向数据库中插入上传文件的信息

        UploadFile uploadFile = new UploadFile();

        // 设置文件分类:音频,图片等等
        uploadFile.setCategory(requiredCategory.getCategory());
        // 设置上传文件的用户
        uploadFile.setUserId(userId);
        // 设置文件创建时间
        uploadFile.setTime(LocalDateTime.now());
        // 设置文件的访问路径
        uploadFile.setUrl(url);

        // 设置文件名
        if(voiceSeedName == null || voiceSeedName.isEmpty()){
            uploadFile.setName(fileName);
        }else {
            uploadFile.setName(voiceSeedName);
        }

        // 写入UploadFile表
        uploadFileMapper.insert(uploadFile);
        log.info("向数据库插入文件成功");

        return uploadFile.getVo();
    }

    @Override
    public PageVo<UploadFileVo> getPageByCategory(UploadFileQuery uploadFileQuery, Long userId) {
        // Assert.notNull 方法会抛出一个 IllegalArgumentException 异常
        Assert.notNull(uploadFileQuery, "用户参数不能为空");

        String fileName = uploadFileQuery.getName();

        //默认按照time降序排序(如果有参数query,就按照参数排序)
        Page<UploadFile> page= uploadFileQuery.toMpPage();

        Page<UploadFile> p = lambdaQuery()
                .eq(UploadFile::getUserId, userId)
                .like(fileName!=null, UploadFile::getName, fileName)
                .page(page);


        //封装成Vo:Po<Po>转为Vo<Vo>
        return PageVo.of(p, UploadFileVo.class);
    }
}

