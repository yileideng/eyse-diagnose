
package com.project.diagnose.controller;

import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.pojo.User;
import com.project.diagnose.dto.query.UserQuery;
import com.project.diagnose.service.LoginService;
import com.project.diagnose.service.UserService;
import com.project.diagnose.dto.vo.PageVo;
import com.project.diagnose.dto.vo.Result;
import com.project.diagnose.dto.vo.UserVo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;

@RestController
@RequestMapping("/user")
public class UserController {
    @Autowired
    private UserService userService;
    @Autowired
    private LoginService loginService;

    //获取当前的登录用户
    @GetMapping("/currentUser")
    //@PreAuthorize("hasAuthority('currentUser')")
    @LogAnnotation(module = "用户操作",operator = "获取当前用户")
    public Result<UserVo> currentUser(@RequestHeader("Authorization") String token){
        return Result.success(loginService.findCurrentUserByToken(token));
    }

    //分页查询:需提供分页参数,注意UserQuery继承了PageQuery.所以分页信息也许提供
    @PostMapping("/page")
    //@PreAuthorize("hasAuthority('getUser')")
    @LogAnnotation(module = "用户操作",operator = "分页查询用户")
    public Result<PageVo<UserVo>> page(@RequestBody UserQuery query){
        PageVo<UserVo> page=userService.getPage(query);
        return Result.success(page);
    }

    //根据id查询用户
    @GetMapping("/page/{id}")
    //@PreAuthorize("hasAuthority('getUser')")
    @LogAnnotation(module = "用户操作",operator = "根据id查询用户")
    public Result<UserVo> getById(@PathVariable Long id){
        UserVo userVo=userService.getUserVoById(id);
        return Result.success(userVo);
    }

    //更新用户信息
    @PutMapping("/update")
    //@PreAuthorize("hasAuthority('updateUser')")
    @LogAnnotation(module = "用户操作",operator = "更新用户")
    public Result<UserVo> update(@RequestBody User user){
        user.setUpdateTime(LocalDateTime.now());
        userService.updateById(user);
        return Result.success(new UserVo(user));
    }

}



