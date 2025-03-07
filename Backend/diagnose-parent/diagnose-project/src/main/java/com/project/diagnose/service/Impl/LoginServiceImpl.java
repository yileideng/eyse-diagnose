

package com.project.diagnose.service.Impl;

import com.baomidou.mybatisplus.core.toolkit.StringUtils;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.diagnose.dto.vo.UserVo;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.LoginUser;
import com.project.diagnose.pojo.User;
import com.project.diagnose.dto.query.LoginQuery;
import com.project.diagnose.service.LoginService;
import com.project.diagnose.service.UserService;
import com.project.diagnose.utils.JWTUtils;
import com.project.diagnose.exception.ErrorMessage;
import com.project.diagnose.dto.vo.LoginVo;
import com.project.diagnose.utils.RedisUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.Objects;

@Service
@Transactional
public class LoginServiceImpl extends ServiceImpl<UserMapper, User> implements LoginService {

    @Autowired
    private UserService userService;
    @Autowired
    private AuthenticationManager authenticationManager;
    @Autowired
    private RedisUtils redisUtils;
    @Autowired
    private PasswordEncoder passwordEncoder;

   /* //password的加密盐
    public static final String salt = "lingrui@#!%&";*/

    // 登录
    @Override
    public LoginVo login(LoginQuery loginQuery) {

        UsernamePasswordAuthenticationToken authenticationToken=new UsernamePasswordAuthenticationToken(loginQuery.getUsername(),loginQuery.getPassword());
        // 用户名密码校验
        Authentication authentication=authenticationManager.authenticate(authenticationToken);

        // 检查是否认证成功
        if(Objects.isNull(authentication)){
            // 实际执行不到这里,在authenticate()认证时,SpringSecurity就会抛出AuthenticationException异常
            throw new DiagnoseException(ErrorMessage.ACCOUNT_PWD_WRONG, HttpStatus.UNAUTHORIZED);
        }

        // 底层是调用UserDetailService获取数据库中的loginUser
        LoginUser loginUser = (LoginUser) authentication.getPrincipal();
        // 根据id创建token
        String token = JWTUtils.createToken(loginUser.getUser().getId());
        // 将(TOKEN_token, loginUser)存入redis, 并设置过期时间为1天
        redisUtils.setLoginUserInRedis(token, loginUser);

        LoginVo loginVo = new LoginVo(loginUser);
        loginVo.setToken(token);
        //loginUserVo.setTokenExpiresIn(TimeUnit.DAYS.toSeconds(1));

        return loginVo;
    }

    //退出登录
    @Override
    public void logout(String token) {
        redisUtils.deleteLoginUserInRedis(token);
    }

    //注册
    @Override
    public void register(LoginQuery loginQuery) {
        String password = loginQuery.getPassword();
        String username = loginQuery.getUsername();
        if (StringUtils.isBlank(password) || StringUtils.isBlank(username)){
            throw new DiagnoseException(ErrorMessage.PARAMS_ERROR, HttpStatus.BAD_REQUEST);
        }
        User user =  userService.findUserByUsername(username);
        if (user != null){
            throw new DiagnoseException(ErrorMessage.ACCOUNT_EXIST, HttpStatus.CONFLICT);
        }
        //如果账号未被注册,则把用户信息添加到数据库
        user = new User();
        user.setUsername(username);
        user.setPassword(passwordEncoder.encode(password));
        user.setUpdateTime(LocalDateTime.now());
        user.setRoleId((long)0);
        user.setEmail(loginQuery.getEmail());
        user.setPhoneNumber(loginQuery.getPhoneNumber());
        user.setAvatarUrl(loginQuery.getAvatarUrl());
        this.userService.save(user);

        //String token = JWTUtils.createToken(user.getId());
        //redisTemplate.opsForValue().set("TOKEN_"+token, JSON.toJSONString(new LoginUser(user,null)),1, TimeUnit.DAYS);
        //return Result.success(token);
    }


    @Override
    public UserVo findCurrentUserByToken(String token) {
        // 检查 token 是否合法
        LoginUser loginUser = checkToken(token);
        if (loginUser == null) {
            // 如果 token 无效，直接抛出异常
            throw new DiagnoseException(ErrorMessage.TOKEN_ERROR, HttpStatus.UNAUTHORIZED);
        }

        UserVo userVo = new UserVo(loginUser);

        return userVo;
    }

    //根据token查找当前登录的用户时:检查token是否合法
    public LoginUser checkToken(String token) {
        if (StringUtils.isBlank(token)){
            return null;
        }
        Map<String, Object> stringObjectMap = JWTUtils.checkToken(token);
        if (stringObjectMap == null){
            return null;
        }

        //根据json查询Redis中的User
        LoginUser loginUser = redisUtils.getLoginUserInRedis(token);

        return loginUser;
    }
}


