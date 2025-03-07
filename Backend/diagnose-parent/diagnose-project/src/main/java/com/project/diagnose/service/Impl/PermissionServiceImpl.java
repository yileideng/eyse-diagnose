package com.project.diagnose.service.Impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.diagnose.mapper.PermissionMapper;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.Permission;
import com.project.diagnose.pojo.User;
import com.project.diagnose.dto.query.PermissionQuery;
import com.project.diagnose.service.PermissionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Service
@Transactional
public class PermissionServiceImpl extends ServiceImpl<PermissionMapper, Permission> implements PermissionService {
    @Autowired
    private PermissionMapper permissionMapper;
    @Autowired
    private UserMapper userMapper;
    //根据用户获取该用户的权限
    @Override
    public List<String> findPermissionPathByUserId(Long id) {
        //根据id获取用户对象
        User user =getUserByUserId(id);
        if(user == null){
            return null;
        }
        //根据用户的角色id查询角色对应的权限
        List<Permission> permissionList = permissionMapper.findPermissionByRoleId(user.getRoleId());
        if(permissionList == null){
            return null;
        }
        //把权限转化成String
        List<String> pathList=new ArrayList<>();
        for(Permission permission:permissionList) {
            pathList.add(permission.getName());
        }
        return pathList;
    }
    //根据userId获取user对象
    private User getUserByUserId(Long id){
        return userMapper.selectById(id);
    }

    @Override
    public void save(PermissionQuery permissionQuery){
        //更新permission表
        Permission permission=new Permission();
        permission.setName(permissionQuery.getPath());
        permissionMapper.insert(permission);

        //获取新插入Permission的id(实际上添加了注解@TableId(type = IdType.AUTO)后,调用MP的insert方法,会自动给实体类对象赋值生成的主键ID)
        Long permissionId=lambdaQuery().eq(Permission::getName,permission.getName()).one().getId();

        //更新role_permission表
        for(Long roleId:permissionQuery.getRoleIds()){
            permissionMapper.insertPermissionRole(roleId,permissionId,permissionQuery.getComment());
        }
    }

    @Override
    public void deletePermission(Long permissionId) {
        permissionMapper.deleteById(permissionId);
        permissionMapper.deletePermissionRole(permissionId);
    }
}
