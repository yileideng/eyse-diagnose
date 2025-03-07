package com.project.diagnose.dto.query;

import com.baomidou.mybatisplus.core.metadata.OrderItem;
import lombok.Data;

@Data
public class UploadFileQuery extends PageQuery{
    private Long id;
    private String name;

    @Override
    protected OrderItem getDefaultOrderItem() {
        // 默认按照时间降序排序
        return new OrderItem("time", false);
    }
}
