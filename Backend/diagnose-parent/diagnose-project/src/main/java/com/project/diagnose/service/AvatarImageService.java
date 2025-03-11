package com.project.diagnose.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.project.diagnose.dto.query.AvatarQuery;
import com.project.diagnose.dto.vo.PageVo;
import com.project.diagnose.dto.vo.AvatarImageVo;
import com.project.diagnose.pojo.AvatarImage;
import com.project.diagnose.utils.FileUtils;
import org.springframework.web.multipart.MultipartFile;


/**
 * <p>
 *  服务类
 * </p>
 *
 * @author itcast
 * @since 2025-03-04
 */
public interface AvatarImageService extends IService<AvatarImage> {
    String uploadAndInsert(String bucket, MultipartFile file, FileUtils.Category requiredCategory, Long userId);
    PageVo<AvatarImageVo> getPageByCategory(AvatarQuery avatarQuery, Long userId);
}
