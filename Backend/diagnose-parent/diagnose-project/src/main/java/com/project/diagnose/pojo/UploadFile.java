package com.project.diagnose.pojo;

import com.baomidou.mybatisplus.annotation.TableName;
import com.project.diagnose.dto.vo.UploadFileVo;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;

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
public class UploadFile implements Serializable {


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


    @NoArgsConstructor
    @AllArgsConstructor
    @Getter
    public enum Category {
        CATEGORY_VOICE("voice"),
        CATEGORY_COURSEWARE("courseware"),
        CATEGORY_VIDEO("video"),
        CATEGORY_AVATAR("avatar");

        private String category;
    }

    public UploadFileVo getVo(){
        return new UploadFileVo(this);
    }
}
