

package com.project.diagnose.controller;

import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.dto.query.LoginQuery;
import com.project.diagnose.dto.query.UserQuery;
import com.project.diagnose.dto.vo.LoginVo;
import com.project.diagnose.service.LoginService;
import com.project.diagnose.dto.vo.Result;

import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping()
public class LoginController {
    @Autowired
    private LoginService loginService;

    @PostMapping("/login")
    @LogAnnotation(module = "登录操作",operator = "用户登录")
    public Result<LoginVo> passwordLogin(@RequestBody LoginQuery loginQuery){
        return Result.success(loginService.passwordLogin(loginQuery));
    }

    @PostMapping("/login-mail")
    @LogAnnotation(module = "登录操作",operator = "用户登录")
    public Result<LoginVo> mailLogin(@RequestBody LoginQuery loginQuery){
        return Result.success(loginService.emailLogin(loginQuery));
    }

    @GetMapping("/mail")
    @LogAnnotation(module = "发送验证码",operator = "用户登录")
    public Result<Boolean> sendMail(@RequestParam String mail){
        loginService.generateMail(mail);
        return Result.success();
    }

    @PostMapping("/register")
    @LogAnnotation(module = "注册操作",operator = "用户注册")
    public Result<Boolean> register(@RequestBody LoginQuery loginQuery){
        loginService.register(loginQuery);
        return Result.success();
    }

    @GetMapping("/user/logout")
    //@PreAuthorize("hasAuthority('logout')")
    @LogAnnotation(module = "退出登录操作",operator = "用户退出登录")
    public Result<Boolean> logout(@RequestHeader("Authorization")String token){
        loginService.logout(token);
        return Result.success();
    }

    // 忘记密码
    @PutMapping("/forget-password")
    //@PreAuthorize("hasAuthority('updateUser')")
    @LogAnnotation(module = "用户操作",operator = "忘记密码")
    public Result<Boolean> forgetPassword(@RequestBody LoginQuery loginQuery){
        loginService.forgetPassword(loginQuery);
        return Result.success(true);
    }

}


