package com.project.diagnose.pojo;

import com.baomidou.mybatisplus.annotation.TableName;
import com.project.diagnose.dto.vo.DiagnoseImageVo;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("diagnose_image")


public class DiagnoseImage {

    /**
     * id
     */
    private Long id;

    /**
     * 名称
     */
    private String name;

    private String storageSource;

    private String bucket;

    private String objectPath;

    private String url;

    /**
     * 上传文件的用户id
     */
    private Long userId;

    private LocalDateTime time;

    private String category;

    private Long reportId;

    public DiagnoseImageVo toVo(){
        return new DiagnoseImageVo(this);
    }

}
