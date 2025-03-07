package com.project.diagnose.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.project.diagnose.dto.query.UploadFileQuery;
import com.project.diagnose.dto.vo.PageVo;
import com.project.diagnose.dto.vo.UploadFileVo;
import com.project.diagnose.pojo.UploadFile;
import org.springframework.web.multipart.MultipartFile;


/**
 * <p>
 *  服务类
 * </p>
 *
 * @author itcast
 * @since 2025-03-04
 */
public interface UploadFileService extends IService<UploadFile> {
    UploadFileVo uploadAndInsert(String bucket, MultipartFile file, UploadFile.Category requiredCategory, Long userId, String voiceSeedName);
    PageVo<UploadFileVo> getPageByCategory(UploadFileQuery uploadFileQuery, Long userId);
}
