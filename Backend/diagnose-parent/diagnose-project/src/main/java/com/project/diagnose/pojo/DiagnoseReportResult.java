package com.project.diagnose.pojo;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("diagnose_report_result")
public class DiagnoseReportResult {
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    private String text;
}
