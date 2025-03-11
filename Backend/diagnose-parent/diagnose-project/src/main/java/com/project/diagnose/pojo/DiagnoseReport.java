package com.project.diagnose.pojo;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

import java.time.LocalDateTime;

@Data
@TableName("diagnose_report")
public class DiagnoseReport {
    /**
     * id
     */
    private Long id;

    private Long userId;

    private LocalDateTime time;

    private Long reportResultId;
}
