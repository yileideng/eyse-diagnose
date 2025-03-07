package com.project.diagnose.dto.query;

import cn.hutool.core.util.StrUtil;
import com.baomidou.mybatisplus.core.metadata.OrderItem;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import lombok.Data;

//分页查询的参数，谁要分页查询，继承PageQuery
@Data
public class PageQuery {
    //页码
    private Integer pageNo = 1;
    //一页的数据个数
    private Integer pageSize =5 ;
    //分页的排序字段
    private String sortBy;
    //是否升序
    private Boolean isAsc = true;

    // 新增方法：获取默认排序条件
    protected OrderItem getDefaultOrderItem() {
        // 默认返回 null，由子类实现具体逻辑
        return null;
    }


    //设置分页查询的条件以及排序规则等
    public <T> Page<T> toMpPage(){
        OrderItem defaultOrderItem = getDefaultOrderItem();
        //分页条件
        Page<T> page=Page.of(pageNo,pageSize);
        //排序条件(默认按照username升序排序)
        if(StrUtil.isNotBlank(sortBy)){
            page.addOrder(new OrderItem(sortBy,isAsc));
        }else if (defaultOrderItem != null) {
            page.addOrder(defaultOrderItem);
        }
        return page;
    }
}
