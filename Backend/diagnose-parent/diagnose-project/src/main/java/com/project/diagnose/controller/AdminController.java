
package com.project.diagnose.controller;

import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.pojo.User;
import com.project.diagnose.dto.query.PermissionQuery;
import com.project.diagnose.service.PermissionService;
import com.project.diagnose.service.UserService;
import com.project.diagnose.dto.vo.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;

@RestController
@RequestMapping("/admins")
public class AdminController {
    @Autowired
    private UserService userService;
    @Autowired
    private PermissionService permissionService;
    @Autowired
    private PasswordEncoder passwordEncoder;

    //管理员删除用户
    @DeleteMapping("/delete/{id}")
    //@PreAuthorize("hasAuthority('chageUser')")
    @LogAnnotation(module = "管理员操作",operator = "删除用户")
    public Result<Boolean> deleteUsers(@PathVariable Long id){
        userService.removeById(id);
        return Result.success();
    }

    //管理员添加用户:管理员可以操作用户任何数据
    @PostMapping("/add")
    //@PreAuthorize("hasAuthority('chageUser')")
    @LogAnnotation(module = "管理员操作",operator = "添加用户")
    public Result<Boolean> addUser(@RequestBody User user){
        user.setUpdateTime(LocalDateTime.now());
        //对password加密(LoginServiceImpl.salt为加密盐)
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        userService.save(user);
        return Result.success();
    }

    //管理员添加权限
    @PostMapping("/Permissions/add")
    //@PreAuthorize("hasAuthority('chagePermission')")
    @LogAnnotation(module = "管理员操作",operator = "添加权限")
    public Result<Boolean> addPermission(@RequestBody PermissionQuery permissionQuery){
        permissionService.save(permissionQuery);
        return Result.success();
    }

    //管理员删除权限
    @DeleteMapping("/Permissions/delete/{permissionId}")
    //@PreAuthorize("hasAuthority('chagePermission')")
    @LogAnnotation(module = "管理员操作",operator = "删除权限")
    public Result<Boolean> deletePermission(@PathVariable Long permissionId){
        permissionService.deletePermission(permissionId);
        return Result.success();
    }


}



