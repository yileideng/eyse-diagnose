package com.project.diagnose.service.Impl;

import cn.hutool.core.lang.Assert;
import com.alibaba.fastjson.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.diagnose.client.MLClient;
import com.project.diagnose.constans.DiagnoseMode;
import com.project.diagnose.dto.query.DiagnoseQuery;
import com.project.diagnose.dto.response.BulkDiagnoseResponse;
import com.project.diagnose.dto.response.PersonalDiagnoseResponse;
import com.project.diagnose.dto.response.UploadFileResponse;
import com.project.diagnose.dto.vo.*;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.mapper.DiagnoseImageMapper;
import com.project.diagnose.mapper.DiagnoseReportMapper;
import com.project.diagnose.mapper.DiagnoseReportResultMapper;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.DiagnoseImage;
import com.project.diagnose.pojo.DiagnoseReport;
import com.project.diagnose.pojo.DiagnoseReportResult;
import com.project.diagnose.pojo.User;
import com.project.diagnose.service.DiagnoseService;
import com.project.diagnose.utils.AliOSSUtils;
import com.project.diagnose.utils.FileUtils;
import com.project.diagnose.utils.MinioUtils;
import com.project.diagnose.utils.UploadFileUtilsFactory;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.util.TempFile;
import org.jetbrains.annotations.NotNull;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
public class DiagnoseServiceImpl extends ServiceImpl<DiagnoseImageMapper, DiagnoseImage> implements DiagnoseService {
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

        DiagnoseImage diagnoseImage = new DiagnoseImage();

        // 设置文件分类:音频,图片等等
        diagnoseImage.setCategory(requiredCategory.getCategory());
        // 设置上传文件的用户
        diagnoseImage.setUserId(userId);
        // 设置文件创建时间
        diagnoseImage.setTime(LocalDateTime.now());
        diagnoseImage.setStorageSource(response.getStorageSource());
        diagnoseImage.setBucket(bucket);
        diagnoseImage.setObjectPath(response.getObjectPath());
        // 设置文件的访问路径
        diagnoseImage.setUrl(response.getUrl());
        // 设置文件名
        diagnoseImage.setName(fileName);

        // 写入UploadFile表
        diagnoseImageMapper.insert(diagnoseImage);
        log.info("向数据库插入文件成功");

        return diagnoseImage.toVo();
    }

    @Override
    public PageVo<DiagnoseReportVo> getDiagnoseHistory(Long userId, DiagnoseQuery diagnoseQuery) {
        // Assert.notNull 方法会抛出一个 IllegalArgumentException 异常
        Assert.notNull(diagnoseQuery, "用户参数不能为空");

        Long id = diagnoseQuery.getId();

        //默认按照username升序排序(如果有参数query,就按照参数排序)
        Page<DiagnoseReport> page=diagnoseQuery.toMpPage();

        //查询条件
        Page<DiagnoseReport> p=diagnoseReportMapper.selectPage(
                page,
                new LambdaQueryWrapper<DiagnoseReport>()
                        .eq(DiagnoseReport::getUserId, userId)
                        .eq(id!=null, DiagnoseReport::getId, id)
        );

        //封装成Vo:Po<Po>转为Vo<Vo>
        return PageVo.of(p, DiagnoseReportVo.class);
    }

    @Override
    public BulkDiagnoseReportResultVo generateBulkDiagnoseReport(Long userId, List<String> idList) {
        // 根据报告查询要诊断的图片数据
        List<DiagnoseImage> diagnoseImageList = diagnoseImageMapper.selectBatchIds(idList);
        List<File> fileList = new ArrayList<>();
        List<String> urlList = new ArrayList<>();

        // 获取诊断图片的文件
        diagnoseImageList.forEach(diagnoseImage -> {
            // 检查文件类型
            if(!FileUtils.checkFileCategory(diagnoseImage.getUrl(), FileUtils.Category.CATEGORY_ZIP)){
                throw new DiagnoseException("文件类型有误，请上传zip压缩包", HttpStatus.BAD_REQUEST);
            }
            File file = downloadFile(diagnoseImage);
            log.info(file.getName());
            fileList.add(file);
            urlList.add(diagnoseImage.getUrl());
        });

        DiagnoseReportResult diagnoseReportResult = new DiagnoseReportResult();
        BulkDiagnoseResponse bulkDiagnoseResponse = null;
        Long resultId;
        try {
            // 发送请求获取诊断结果
            bulkDiagnoseResponse = mlClient.requestForBulkDiagnose(fileList);
            diagnoseReportResult.setText(JSON.toJSONString(bulkDiagnoseResponse));
            // 将诊断结果数据以JSON格式存入数据库
            diagnoseReportResultMapper.insert(diagnoseReportResult);
            resultId = diagnoseReportResult.getId();

        } catch (IOException e) {
            log.info("发送请求，获取诊断结果失败: {}", e.getMessage());
            throw new DiagnoseException("获取诊断结果失败");
        }

        // 插入诊断报告的数据
        DiagnoseReport diagnoseReport = new DiagnoseReport();
        diagnoseReport.setTime(LocalDateTime.now());
        diagnoseReport.setUserId(userId);
        diagnoseReport.setDiagnoseMode(DiagnoseMode.BULK.getValue());
        // 关联诊断结果
        diagnoseReport.setReportResultId(resultId);
        diagnoseReportMapper.insert(diagnoseReport);
        Long diagnoseId = diagnoseReport.getId();

        // 添加诊断图片对应的诊断报告id
        diagnoseImageList.forEach(diagnoseImage -> {
            diagnoseImage.setReportId(diagnoseId);
            diagnoseImageMapper.updateById(diagnoseImage);
        });

        // 构造返回结果
        BulkDiagnoseReportResultVo bulkDiagnoseReportResultVo = buildBulkDiagnoseReportResultVo(userId, diagnoseId, DiagnoseMode.BULK, bulkDiagnoseResponse, urlList);

        return bulkDiagnoseReportResultVo;
    }

    private File downloadFile(DiagnoseImage diagnoseImage) {
        try (InputStream inputStream = uploadFileUtilsFactory.download(diagnoseImage)) {
            File minioFile = TempFile.createTempFile("minio", ".tmp");
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



    @Override
    public BulkDiagnoseReportResultVo getBulkDiagnoseDetails(Long userId, Long diagnoseId) {
        // 查询诊断报告
        DiagnoseReport diagnoseReport = diagnoseReportMapper.selectOne(new LambdaQueryWrapper<DiagnoseReport>()
                .eq(DiagnoseReport::getId, diagnoseId)
                .eq(DiagnoseReport::getUserId, userId)
        );
        if(diagnoseReport==null){
            throw new DiagnoseException("该诊断报告不存在", HttpStatus.NOT_FOUND);
        }
        // 校验报告类型为bulk
        if(!DiagnoseMode.BULK.getValue().equals(diagnoseReport.getDiagnoseMode())){
            throw new DiagnoseException("该报告不是批量诊断报告", HttpStatus.BAD_REQUEST);
        }
        // 查询诊断报告关联的图片
        List<DiagnoseImage> diagnoseImageList = diagnoseImageMapper.selectList(new LambdaQueryWrapper<DiagnoseImage>().eq(DiagnoseImage::getReportId, diagnoseReport.getId()));
        List<String> urlList = diagnoseImageList.stream().map(DiagnoseImage::getUrl).collect(Collectors.toList());
        // 查询诊断结果
        DiagnoseReportResult diagnoseReportResult = diagnoseReportResultMapper.selectById(diagnoseReport.getReportResultId());
        BulkDiagnoseResponse bulkDiagnoseResponse = JSON.parseObject(diagnoseReportResult.getText(), BulkDiagnoseResponse.class);
        // 查询诊断用户
        User user = userMapper.selectById(userId);

        // 构造返回的结果
        BulkDiagnoseReportResultVo bulkDiagnoseReportResultVo = buildBulkDiagnoseReportResultVo(userId, diagnoseId, DiagnoseMode.BULK, bulkDiagnoseResponse, urlList);
        return bulkDiagnoseReportResultVo;
    }

    @Override
    public BulkDiagnoseReportResultVo generatePersonalDiagnoseReport(Long userId, List<String> idList) {
        // 根据报告查询要诊断的图片数据
        List<DiagnoseImage> diagnoseImageList = diagnoseImageMapper.selectBatchIds(idList);
        List<File> fileList = new ArrayList<>();
        List<String> urlList = new ArrayList<>();

        // 获取诊断图片的文件
        diagnoseImageList.forEach(diagnoseImage -> {
            // 检查文件类型
            if(!FileUtils.checkFileCategory(diagnoseImage.getUrl(), FileUtils.Category.CATEGORY_IMAGE)){
                throw new DiagnoseException("文件类型有误，请上传图片", HttpStatus.BAD_REQUEST);
            }
            File file = downloadFile(diagnoseImage);
            log.info(file.getName());
            fileList.add(file);
            urlList.add(diagnoseImage.getUrl());
        });

        DiagnoseReportResult diagnoseReportResult = new DiagnoseReportResult();
        BulkDiagnoseResponse bulkDiagnoseResponse = new BulkDiagnoseResponse();
        Long resultId;
        try {
            // 发送请求获取诊断结果
            bulkDiagnoseResponse = mlClient.requestForPersonalDiagnose(fileList);
            diagnoseReportResult.setText(JSON.toJSONString(bulkDiagnoseResponse));
            // 将诊断结果数据以JSON格式存入数据库
            diagnoseReportResultMapper.insert(diagnoseReportResult);
            resultId = diagnoseReportResult.getId();

        } catch (IOException e) {
            log.info("发送请求，获取诊断结果失败: {}", e.getMessage());
            throw new DiagnoseException("获取诊断结果失败");
        }

        // 插入诊断报告的数据
        DiagnoseReport diagnoseReport = new DiagnoseReport();
        diagnoseReport.setTime(LocalDateTime.now());
        diagnoseReport.setUserId(userId);
        diagnoseReport.setDiagnoseMode(DiagnoseMode.PERSONAL.getValue());
        // 关联诊断结果
        diagnoseReport.setReportResultId(resultId);
        diagnoseReportMapper.insert(diagnoseReport);
        Long diagnoseId = diagnoseReport.getId();

        // 添加诊断图片对应的诊断报告id
        diagnoseImageList.forEach(diagnoseImage -> {
            diagnoseImage.setReportId(diagnoseId);
            diagnoseImageMapper.updateById(diagnoseImage);
        });

        // 构造返回结果
        BulkDiagnoseReportResultVo bulkDiagnoseReportResultVo = buildBulkDiagnoseReportResultVo(userId, diagnoseId, DiagnoseMode.PERSONAL, bulkDiagnoseResponse, urlList);

        return bulkDiagnoseReportResultVo;
    }

    @Override
    public BulkDiagnoseReportResultVo getPersonalDiagnoseDetails(Long userId, Long diagnoseId) {
        // 查询诊断报告
        DiagnoseReport diagnoseReport = diagnoseReportMapper.selectOne(new LambdaQueryWrapper<DiagnoseReport>()
                .eq(DiagnoseReport::getId, diagnoseId)
                .eq(DiagnoseReport::getUserId, userId)
        );
        if(diagnoseReport==null){
            throw new DiagnoseException("该诊断报告不存在", HttpStatus.NOT_FOUND);
        }
        // 校验报告类型为personal
        if(!DiagnoseMode.PERSONAL.getValue().equals(diagnoseReport.getDiagnoseMode())){
            throw new DiagnoseException("该报告不是个人诊断报告", HttpStatus.BAD_REQUEST);
        }
        // 查询诊断报告关联的图片
        List<DiagnoseImage> diagnoseImageList = diagnoseImageMapper.selectList(new LambdaQueryWrapper<DiagnoseImage>().eq(DiagnoseImage::getReportId, diagnoseReport.getId()));
        List<String> urlList = diagnoseImageList.stream().map(DiagnoseImage::getUrl).collect(Collectors.toList());
        // 查询诊断结果
        DiagnoseReportResult diagnoseReportResult = diagnoseReportResultMapper.selectById(diagnoseReport.getReportResultId());
        BulkDiagnoseResponse bulkDiagnoseResponse = JSON.parseObject(diagnoseReportResult.getText(), BulkDiagnoseResponse.class);

        // 构造返回结果
        BulkDiagnoseReportResultVo bulkDiagnoseReportResultVo = buildBulkDiagnoseReportResultVo(userId, diagnoseId, DiagnoseMode.PERSONAL, bulkDiagnoseResponse, urlList);
        return bulkDiagnoseReportResultVo;
    }

    @NotNull
    private BulkDiagnoseReportResultVo buildBulkDiagnoseReportResultVo(Long userId, Long diagnoseId, DiagnoseMode mode, BulkDiagnoseResponse bulkDiagnoseResponse, List<String> urlList) {
        // 返回结果
        BulkDiagnoseReportResultVo bulkDiagnoseReportResultVo = new BulkDiagnoseReportResultVo();
        // 构建报告基本信息
        bulkDiagnoseReportResultVo.setId(diagnoseId.toString());
        bulkDiagnoseReportResultVo.setTime(LocalDateTime.now().toString());
        // 构建用户基本信息
        User user = userMapper.selectById(userId);
        bulkDiagnoseReportResultVo.setUserId(user.getId().toString());
        bulkDiagnoseReportResultVo.setUsername(user.getUsername());
        bulkDiagnoseReportResultVo.setEmail(user.getEmail());
        bulkDiagnoseReportResultVo.setPhoneNumber(user.getPhoneNumber());
        bulkDiagnoseReportResultVo.setAvatarUrl(user.getAvatarUrl());
        // 构建报告结果信息
        bulkDiagnoseReportResultVo.setReport(bulkDiagnoseResponse.toBulkDiagnoseResponseList(mode));
        bulkDiagnoseReportResultVo.setUrlList(urlList);
        return bulkDiagnoseReportResultVo;
    }

}
