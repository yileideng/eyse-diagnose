package com.project.diagnose.service.Impl;

import cn.hutool.core.lang.Assert;
import com.alibaba.fastjson.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.StringUtils;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.diagnose.client.MLClient;
import com.project.diagnose.constans.DiagnoseMode;
import com.project.diagnose.dto.query.DiagnoseQuery;
import com.project.diagnose.dto.response.DiagnoseResponse;
import com.project.diagnose.dto.response.DiagnoseResponseList;
import com.project.diagnose.dto.response.UploadFileResponse;
import com.project.diagnose.dto.vo.*;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.mapper.DiagnoseImageMapper;
import com.project.diagnose.mapper.DiagnoseReportMapper;
import com.project.diagnose.mapper.DiagnoseReportResultMapper;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.DiagnoseFile;
import com.project.diagnose.pojo.DiagnoseBase;
import com.project.diagnose.pojo.DiagnoseResult;
import com.project.diagnose.pojo.User;
import com.project.diagnose.service.DiagnoseService;
import com.project.diagnose.utils.AliOSSUtils;
import com.project.diagnose.utils.FileUtils;
import com.project.diagnose.utils.MinioUtils;
import com.project.diagnose.utils.UploadFileUtilsFactory;
import lombok.extern.slf4j.Slf4j;
import org.apache.logging.log4j.util.StringBuilders;
import org.apache.poi.util.TempFile;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
public class DiagnoseServiceImpl extends ServiceImpl<DiagnoseImageMapper, DiagnoseFile> implements DiagnoseService {
    @Autowired
    private MinioUtils minioUtils;
    @Autowired
    private AliOSSUtils aliOSSUtils;
    @Autowired
    private DiagnoseImageMapper  diagnoseImageMapper;
    @Autowired
    private DiagnoseReportMapper diagnoseReportMapper;
    @Autowired
    private UserMapper userMapper;
    @Autowired
    private MLClient mlClient;
    @Autowired
    private DiagnoseReportResultMapper diagnoseReportResultMapper;
    @Autowired
    private UploadFileUtilsFactory uploadFileUtilsFactory;

    @Override
    public List<DiagnoseImageVo> uploadFiles(String bucket, MultipartFile[] files, FileUtils.Category requiredCategory, Long userId){
        List<DiagnoseImageVo> diagnoseImageVos = new ArrayList<>();
        for (MultipartFile file : files) {
            DiagnoseImageVo diagnoseImageVo = uploadFile(bucket, file, requiredCategory, userId);
            diagnoseImageVos.add(diagnoseImageVo);
        }
        return diagnoseImageVos;
    }

    private DiagnoseImageVo uploadFile(String bucket, MultipartFile file, FileUtils.Category requiredCategory, Long userId){

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

        DiagnoseFile diagnoseFile = new DiagnoseFile();

        // 设置文件分类:音频,图片等等
        diagnoseFile.setCategory(requiredCategory.getCategory());
        // 设置上传文件的用户
        diagnoseFile.setUserId(userId);
        // 设置文件创建时间
        diagnoseFile.setTime(LocalDateTime.now());
        diagnoseFile.setStorageSource(response.getStorageSource());
        diagnoseFile.setBucket(bucket);
        diagnoseFile.setObjectPath(response.getObjectPath());
        // 设置文件的访问路径
        diagnoseFile.setUrl(response.getUrl());
        // 设置文件名
        diagnoseFile.setName(fileName);

        // 写入UploadFile表
        diagnoseImageMapper.insert(diagnoseFile);
        log.info("向数据库插入文件成功");

        return diagnoseFile.toVo();
    }



    @Override
    public DiagnoseResponseVo generateBulkDiagnoseReport(Long userId, List<String> idList) {
        // 诊断模式：批量
        DiagnoseMode diagnoseMode = DiagnoseMode.BULK;

        // 根据报告查询要诊断的图片数据
        List<DiagnoseFile> diagnoseFileList = diagnoseImageMapper.selectBatchIds(idList);
        List<File> fileList = new ArrayList<>();
        List<String> urlList = new ArrayList<>();

        // 获取诊断图片的文件
        diagnoseFileList.forEach(diagnoseImage -> {
            // 检查文件类型
            if(!FileUtils.checkFileCategory(diagnoseImage.getUrl(), FileUtils.Category.CATEGORY_ZIP)){
                throw new DiagnoseException("文件类型有误，请上传zip压缩包", HttpStatus.BAD_REQUEST);
            }
            File file = downloadFile(diagnoseImage);
            log.info(file.getName());
            fileList.add(file);
            urlList.add(diagnoseImage.getUrl());
        });

        DiagnoseResult diagnoseResult = new DiagnoseResult();
        DiagnoseResponseList diagnoseResponseList;
        Long resultId;
        try {
            // 发送请求获取诊断结果
            DiagnoseResponse diagnoseResponse = mlClient.requestForPersonalDiagnose(fileList);
            // 将诊断结果从Map转为List
            diagnoseResponseList = diagnoseResponse.toBulkDiagnoseResponseList(diagnoseMode.getValue());
            // 将诊断结果数据以JSON格式存入数据库
            diagnoseResult.setText(JSON.toJSONString(diagnoseResponseList));
            diagnoseReportResultMapper.insert(diagnoseResult);
            resultId = diagnoseResult.getId();

        } catch (IOException e) {
            log.info("发送请求，获取诊断结果失败: {}", e.getMessage());
            throw new DiagnoseException("获取诊断结果失败");
        }

        // 插入诊断报告的数据
        DiagnoseBase diagnoseBase = new DiagnoseBase();
        diagnoseBase.setTime(LocalDateTime.now());
        diagnoseBase.setUserId(userId);
        diagnoseBase.setDiagnoseMode(diagnoseMode.getValue());
        // 关联诊断结果
        diagnoseBase.setReportResultId(resultId);
        diagnoseReportMapper.insert(diagnoseBase);
        Long diagnoseId = diagnoseBase.getId();

        // 添加诊断图片对应的诊断报告id
        diagnoseFileList.forEach(diagnoseImage -> {
            diagnoseImage.setReportId(diagnoseId);
            diagnoseImageMapper.updateById(diagnoseImage);
        });

        // 构造返回结果
        DiagnoseResponseVo diagnoseResponseVo = buildBulkDiagnoseReportResultVo(userId, diagnoseId, diagnoseBase.getDiagnoseMode(), diagnoseResponseList, urlList);

        return diagnoseResponseVo;
    }


    @Override
    public DiagnoseResponseVo generatePersonalDiagnoseReport(Long userId, List<String> idList) {
        // 诊断模式：个人
        DiagnoseMode diagnoseMode = DiagnoseMode.PERSONAL;

        // 根据报告查询要诊断的图片数据
        List<DiagnoseFile> diagnoseFileList = diagnoseImageMapper.selectBatchIds(idList);
        List<File> fileList = new ArrayList<>();
        List<String> urlList = new ArrayList<>();

        // 获取诊断图片的文件
        diagnoseFileList.forEach(diagnoseImage -> {
            // 检查文件类型
            if(!FileUtils.checkFileCategory(diagnoseImage.getUrl(), FileUtils.Category.CATEGORY_IMAGE)){
                throw new DiagnoseException("文件类型有误，请上传图片", HttpStatus.BAD_REQUEST);
            }
            File file = downloadFile(diagnoseImage);
            log.info(file.getName());
            fileList.add(file);
            urlList.add(diagnoseImage.getUrl());
        });

        DiagnoseResult diagnoseResult = new DiagnoseResult();
        DiagnoseResponseList diagnoseResponseList;
        Long resultId;
        try {
            // 发送请求获取诊断结果
            DiagnoseResponse diagnoseResponse = mlClient.requestForPersonalDiagnose(fileList);
            // 将诊断结果从Map转为List
            diagnoseResponseList = diagnoseResponse.toBulkDiagnoseResponseList(diagnoseMode.getValue());
            // 将诊断结果数据以JSON格式存入数据库
            diagnoseResult.setText(JSON.toJSONString(diagnoseResponseList));
            diagnoseReportResultMapper.insert(diagnoseResult);
            resultId = diagnoseResult.getId();

        } catch (IOException e) {
            log.info("发送请求，获取诊断结果失败: {}", e.getMessage());
            throw new DiagnoseException("获取诊断结果失败");
        }

        // 插入诊断报告的数据
        DiagnoseBase diagnoseBase = new DiagnoseBase();
        diagnoseBase.setTime(LocalDateTime.now());
        diagnoseBase.setUserId(userId);
        diagnoseBase.setDiagnoseMode(diagnoseMode.getValue());
        // 关联诊断结果
        diagnoseBase.setReportResultId(resultId);
        diagnoseReportMapper.insert(diagnoseBase);
        Long diagnoseId = diagnoseBase.getId();

        // 添加诊断图片对应的诊断报告id
        diagnoseFileList.forEach(diagnoseImage -> {
            diagnoseImage.setReportId(diagnoseId);
            diagnoseImageMapper.updateById(diagnoseImage);
        });

        // 构造返回结果
        DiagnoseResponseVo diagnoseResponseVo = buildBulkDiagnoseReportResultVo(userId, diagnoseId, diagnoseBase.getDiagnoseMode(), diagnoseResponseList, urlList);

        return diagnoseResponseVo;
    }
    // 下载文件
    private File downloadFile(DiagnoseFile diagnoseFile) {
        try (InputStream inputStream = uploadFileUtilsFactory.download(diagnoseFile)) {
            StringBuilder stringBuilder = new StringBuilder(diagnoseFile.getName());
            File minioFile = TempFile.createTempFile("minio", ".zip");
            try (OutputStream out = new FileOutputStream(minioFile)) {
                byte[] buffer = new byte[1024];
                int length;
                while ((length = inputStream.read(buffer)) > 0) {
                    out.write(buffer, 0, length);
                }
            }
            return minioFile;
        }catch (Exception e){
            log.info("下载诊断图片失败: {}", e.getMessage());
            throw new DiagnoseException("下载诊断图片失败");
        }
    }

    // 查看诊断历史
    @Override
    public PageVo<DiagnoseBaseVo> getDiagnoseHistory(Long userId, DiagnoseQuery diagnoseQuery) {
        // Assert.notNull 方法会抛出一个 IllegalArgumentException 异常
        Assert.notNull(diagnoseQuery, "用户参数不能为空");

        Long id = diagnoseQuery.getId();

        //默认按照username升序排序(如果有参数query,就按照参数排序)
        Page<DiagnoseBase> page=diagnoseQuery.toMpPage();

        //查询条件
        Page<DiagnoseBase> p=diagnoseReportMapper.selectPage(
                page,
                new LambdaQueryWrapper<DiagnoseBase>()
                        .eq(DiagnoseBase::getUserId, userId)
                        .eq(id!=null, DiagnoseBase::getId, id)
        );

        //封装成Vo:Po<Po>转为Vo<Vo>
        return PageVo.of(p, DiagnoseBaseVo.class);
    }

    // 查看报告详情
    @Override
    public DiagnoseResponseVo getDiagnoseDetails(Long userId, Long diagnoseId) {
        // 查询诊断报告
        DiagnoseBase diagnoseBase = diagnoseReportMapper.selectOne(new LambdaQueryWrapper<DiagnoseBase>()
                .eq(DiagnoseBase::getId, diagnoseId)
                .eq(DiagnoseBase::getUserId, userId)
        );
        if(diagnoseBase ==null){
            throw new DiagnoseException("该诊断报告不存在", HttpStatus.NOT_FOUND);
        }

        // 查询诊断报告关联的图片
        List<DiagnoseFile> diagnoseFileList = diagnoseImageMapper.selectList(new LambdaQueryWrapper<DiagnoseFile>().eq(DiagnoseFile::getReportId, diagnoseBase.getId()));
        List<String> urlList = diagnoseFileList.stream().map(DiagnoseFile::getUrl).collect(Collectors.toList());
        // 查询诊断结果
        DiagnoseResult diagnoseResult = diagnoseReportResultMapper.selectById(diagnoseBase.getReportResultId());
        DiagnoseResponseList diagnoseResponseList = JSON.parseObject(diagnoseResult.getText(), DiagnoseResponseList.class);
        // 查询诊断用户
        User user = userMapper.selectById(userId);

        // 构造返回的结果
        DiagnoseResponseVo diagnoseResponseVo = buildBulkDiagnoseReportResultVo(userId, diagnoseId, diagnoseBase.getDiagnoseMode(), diagnoseResponseList, urlList);
        return diagnoseResponseVo;
    }
    // 构造返回结果
    private DiagnoseResponseVo buildBulkDiagnoseReportResultVo(Long userId, Long diagnoseId, String mode, DiagnoseResponseList diagnoseResponseList, List<String> urlList) {
        // 返回结果
        DiagnoseResponseVo diagnoseResponseVo = new DiagnoseResponseVo();
        // 构建报告基本信息
        diagnoseResponseVo.setId(diagnoseId.toString());
        diagnoseResponseVo.setTime(LocalDateTime.now().toString());
        // 构建用户基本信息
        User user = userMapper.selectById(userId);
        diagnoseResponseVo.setUserId(user.getId().toString());
        diagnoseResponseVo.setUsername(user.getUsername());
        diagnoseResponseVo.setEmail(user.getEmail());
        diagnoseResponseVo.setPhoneNumber(user.getPhoneNumber());
        diagnoseResponseVo.setAvatarUrl(user.getAvatarUrl());
        // 构建报告结果信息
        diagnoseResponseVo.setReport(diagnoseResponseList);
        diagnoseResponseVo.setUrlList(urlList);
        return diagnoseResponseVo;
    }

    private void a(DiagnoseResponse.PredictionResult predictionResult){
        Map<String, DiagnoseResponse.DiseaseInfo> diseases = predictionResult.getDiseases();
    }
}
