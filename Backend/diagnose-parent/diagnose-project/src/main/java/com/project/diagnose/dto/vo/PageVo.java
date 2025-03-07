package com.project.diagnose.dto.vo;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.collection.CollUtil;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import lombok.Data;

import java.util.Collections;
import java.util.List;

@Data
public class PageVo<T> {
    private Long total;
    private Long pages;
    private List<T> list;

    //将Po转为Vo
    public static <Po,Vo> PageVo<Vo> of(Page<Po> p,Class<Vo> voClass){
        //封装Vo
        PageVo<Vo> vo=new PageVo<>();
        vo.setTotal(p.getTotal());
        vo.setPages(p.getPages());
        List<Po> list=p.getRecords();
        //如果集合为空,返回空集合
        if(CollUtil.isEmpty(list)){
            vo.setList(Collections.emptyList());
        }
        //否则把List中的User拷贝给UserVo
        else {
            vo.setList(BeanUtil.copyToList(list, voClass));
        }
        return vo;
    }
}
