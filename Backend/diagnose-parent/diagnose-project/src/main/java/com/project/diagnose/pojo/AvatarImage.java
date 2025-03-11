package com.project.diagnose.pojo;

import com.baomidou.mybatisplus.annotation.TableName;
import com.project.diagnose.dto.vo.AvatarImageVo;
import lombok.Data;

import java.io.Serializable;
import java.time.LocalDateTime;

/**
 * <p>
 * 
 * </p>
 *
 * @author itcast
 */
@Data
@TableName("upload_file")
public class AvatarImage implements Serializable {


    private static final long serialVersionUID = 1L;

    /**
     * id
     */
    private Long id;

    /**
     * 名称
     */
    private String name;


    private String url;

    /**
     * 上传文件的用户id
     */
    private Long userId;

    private LocalDateTime time;

    private String category;


    public AvatarImageVo getVo(){
        return new AvatarImageVo(this);
    }
}
