package com.project.diagnose.dto.query;

import com.baomidou.mybatisplus.core.metadata.OrderItem;
import lombok.Data;


//从前端获取的分页查询的参数
@Data
public class UserQuery extends PageQuery{
    private Long id;

    private String username;

    private String email;

    private String phoneNumber;

    private String avatarUrl;

    private Long roleId;

    @Override
    protected OrderItem getDefaultOrderItem() {
        // 默认返回 null，由子类实现具体逻辑
        return new OrderItem("username",true);
    }
}
