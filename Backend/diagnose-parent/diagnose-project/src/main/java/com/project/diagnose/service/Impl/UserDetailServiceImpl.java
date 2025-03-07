package com.project.diagnose.service.Impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.LoginUser;
import com.project.diagnose.pojo.User;
import com.project.diagnose.service.PermissionService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Objects;

@Slf4j
@Service
public class UserDetailServiceImpl implements UserDetailsService {
    @Autowired
    private UserMapper userMapper;
    @Autowired
    private PermissionService permissionService;
    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        LambdaQueryWrapper<User> lambdaQueryWrapper=new LambdaQueryWrapper<>();
        lambdaQueryWrapper.eq(User::getUsername,username);
        User user=userMapper.selectOne(lambdaQueryWrapper);
        if(Objects.isNull(user)){
            throw new UsernameNotFoundException("登录失败，用户名不存在");
        }

        //获取登录用户的权限列表
        List<String> permissionList = permissionService.findPermissionPathByUserId(user.getId());
        log.info("用户权限列表：{}",permissionList);
        return new LoginUser(user,permissionList);
    }
}
