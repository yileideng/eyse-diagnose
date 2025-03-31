
package com.project.diagnose.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.project.diagnose.dto.query.UserQuery;
import com.project.diagnose.dto.vo.UserVo;
import com.project.diagnose.pojo.User;
import com.project.diagnose.dto.query.LoginQuery;
import com.project.diagnose.dto.vo.LoginVo;

public interface LoginService extends IService<User> {
    //登录
    LoginVo passwordLogin(LoginQuery loginQuery);

    LoginVo emailLogin(LoginQuery loginQuery);

    // 发送邮件
    String generateMail(String mail);

    //退出登录
    void logout(String token);

    //注册
    void register(LoginQuery loginQuery);

    //根据token查询当前登录的用户,把checkToken得到的User封装成Vo返回给前端
    UserVo findCurrentUserByToken(String token);

    void forgetPassword(LoginQuery loginQuery);
}

