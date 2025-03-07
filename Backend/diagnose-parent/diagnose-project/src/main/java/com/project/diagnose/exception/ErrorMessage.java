package com.project.diagnose.exception;

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;


@NoArgsConstructor
@AllArgsConstructor
public enum ErrorMessage {
    ACCOUNT_PWD_WRONG("用户名或密码错误"),
    TOKEN_ERROR("token不合法"),
    ACCOUNT_EXIST("账号已存在"),
    PARAMS_ERROR("请求参数有误"),
    NO_PERMISSION("无访问权限"),
    SESSION_TIME_OUT("会话超时"),
    NO_LOGIN("未登录"),
    SYSTEM_ERROR("系统错误"),;


    private String msg;

    public String getMsg() {
        return msg;
    }
}

