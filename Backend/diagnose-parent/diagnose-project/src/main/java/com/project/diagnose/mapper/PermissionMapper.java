package com.project.diagnose.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.project.diagnose.pojo.Permission;
import org.apache.ibatis.annotations.Delete;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface PermissionMapper extends BaseMapper<Permission> {
    //多表联查:根据roleId查询权限列表
    @Select("select * from permission where id in (select permission_id from role_permission where role_id =#{id})")
    List<Permission>findPermissionByRoleId(Long id);

    //role_permission关联表新增
    @Insert("insert into role_permission (role_id, permission_id ,comment) values (#{roleId}, #{permissionId} ,#{comment})")
    void insertPermissionRole(Long roleId,Long permissionId,String comment);

    //根据permissionId删除关联表
    @Delete("delete from role_permission where permission_id = #{permissionId}")
    void deletePermissionRole(Long permissionId);
}
