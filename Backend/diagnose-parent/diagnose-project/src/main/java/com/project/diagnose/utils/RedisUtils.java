package com.project.diagnose.utils;

import com.alibaba.fastjson.JSON;
import com.project.diagnose.pojo.LoginUser;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Component;

import java.util.concurrent.TimeUnit;

@Component
public class RedisUtils {
    @Autowired
    private RedisTemplate<String,String> redisTemplate;

    public LoginUser getLoginUserInRedis(String token){
        // 从redis中根据token获取loginUser
        String userJson = redisTemplate.opsForValue().get("TOKEN_" + token);
        LoginUser loginUser = JSON.parseObject(userJson, LoginUser.class);//json格式的User转成实体类
        return loginUser;
    }

    public void deleteLoginUserInRedis(String token){
        redisTemplate.delete("TOKEN_"+token);
    }

    public void setLoginUserInRedis(String token,LoginUser loginUser){
        redisTemplate.opsForValue().set("TOKEN_"+token, JSON.toJSONString(loginUser),1, TimeUnit.DAYS);
    }
}
