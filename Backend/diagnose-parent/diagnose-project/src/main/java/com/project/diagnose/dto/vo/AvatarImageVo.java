package com.project.diagnose.dto.vo;

import com.project.diagnose.pojo.AvatarImage;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@NoArgsConstructor
@Data
public class AvatarImageVo {
    /**
     * 文件id
     */
    private String id;

    /**
     * 文件名称
     */
    private String name;

    /**
     * 文件访问路径
     */
    private String url;

    /**
     * 上传文件的用户id
     */
    private String userId;

    private LocalDateTime time;

    private String category;

    public AvatarImageVo(AvatarImage avatarImage) {
        this.setId(avatarImage.getId().toString());
        this.setName(avatarImage.getName());
        this.setUrl(avatarImage.getUrl());
        this.setUserId(avatarImage.getUserId().toString());
        this.setTime(avatarImage.getTime());
        this.setCategory(avatarImage.getCategory());
    }
}
