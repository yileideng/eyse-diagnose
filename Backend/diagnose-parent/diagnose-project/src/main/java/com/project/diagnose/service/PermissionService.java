package com.project.diagnose.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.project.diagnose.pojo.Permission;
import com.project.diagnose.dto.query.PermissionQuery;

import java.util.List;

public interface PermissionService extends IService<Permission> {
    //获取用户的权限列表
    List<String> findPermissionPathByUserId(Long id);

    //添加权限
    void save(PermissionQuery permissionQuery);

    //删除权限
    void deletePermission(Long permissionId);
}
