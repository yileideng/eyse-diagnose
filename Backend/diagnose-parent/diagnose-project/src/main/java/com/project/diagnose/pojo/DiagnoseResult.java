package com.project.diagnose.pojo;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("diagnose_result")
public class DiagnoseResult {
    @TableId(value = "id", type = IdType.AUTO)
    private Long id;

    private String text;
}
