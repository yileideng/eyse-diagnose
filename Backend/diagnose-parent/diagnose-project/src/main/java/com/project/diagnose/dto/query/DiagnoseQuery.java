package com.project.diagnose.dto.query;

import com.baomidou.mybatisplus.core.metadata.OrderItem;
import lombok.Data;

@Data
public class DiagnoseQuery extends PageQuery{
    private Long id;

    protected OrderItem getDefaultOrderItem(){
        return new OrderItem("time", false);
    }
}
