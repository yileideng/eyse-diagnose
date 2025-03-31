package com.project.diagnose.service.Impl;

import cn.hutool.core.lang.Assert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.User;
import com.project.diagnose.dto.query.UserQuery;
import com.project.diagnose.service.LoginService;
import com.project.diagnose.service.UserService;
import com.project.diagnose.dto.vo.PageVo;
import com.project.diagnose.dto.vo.UserVo;
import com.project.diagnose.utils.RedisUtils;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.regex.Pattern;

@Service
@Transactional
public class UserServiceImpl extends ServiceImpl<UserMapper,User> implements UserService{
    @Autowired
    UserMapper userMapper;
    @Autowired
    private RedisUtils redisUtils;
    @Autowired
    private PasswordEncoder passwordEncoder;

    //分页查询
    @Override
    public PageVo<UserVo> getPage(UserQuery userQuery){
        // Assert.notNull 方法会抛出一个 IllegalArgumentException 异常
        Assert.notNull(userQuery, "用户参数不能为空");

        Long id=userQuery.getId();
        String username=userQuery.getUsername();
        Long roleId=userQuery.getRoleId();

        //默认按照username升序排序(如果有参数query,就按照参数排序)
        Page<User> page=userQuery.toMpPage();

        //查询条件
        Page<User> p=lambdaQuery()
                .eq(id!=null,User::getId,id)
                .like(username!=null,User::getUsername,username)
                .eq(roleId!=null,User::getRoleId,roleId)
                .page(page);

        //封装成Vo:Po<Po>转为Vo<Vo>
        return PageVo.of(p,UserVo.class);
    }

    //根据账号密码查询用户
    @Override
    public User findUser(String username, String password) {
        //查找条件
        return lambdaQuery().eq(User::getUsername,username)
                .eq(User::getPassword,password)
                .select(User::getUsername,User::getId)
                .last("limit 1")//查找到了就停止
                .one();//返回一个User对象
    }

    @Override
    public User findUserByUsername(String username) {
        return lambdaQuery().eq(User::getUsername,username).one();
    }

    @Override
    public UserVo getUserVoById(Long id){
        UserVo userVo=new UserVo();
        BeanUtils.copyProperties(userMapper.selectById(id),userVo);
        return userVo;
    }

    @Override
    public void updateUser(Long userId, UserQuery userQuery) {
        UpdateWrapper<User> updateWrapper = new UpdateWrapper<>();
        updateWrapper.eq("id", userId);

        User user = new User();
        user.setUsername(userQuery.getUsername());
        user.setEmail(userQuery.getEmail());
        user.setPhoneNumber(userQuery.getPhoneNumber());
        user.setAvatarUrl(userQuery.getAvatarUrl());

        // MyBatis-Plus会自动忽略null值字段
        userMapper.update(user, updateWrapper);
    }

    @Override
    public User findUserByMail(String mail) {
        LambdaQueryWrapper<User> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.eq(User::getEmail,mail);
        return userMapper.selectOne(queryWrapper);
    }

    @Override
    public void updatePassword(Long userId, UserQuery userQuery) {
        String oldPassword = userQuery.getOldPassword();
        String newPassword = userQuery.getNewPassword();
        String confirmPassword = userQuery.getConfirmPassword();
        if (oldPassword == null || newPassword == null || confirmPassword == null) {
            throw new DiagnoseException("密码字段不能为空");
        }
        if (!newPassword.equals(confirmPassword)) {
            throw new DiagnoseException("两次输入的新密码不一致", HttpStatus.BAD_REQUEST);
        }
        validatePassword(newPassword);

        User user = userMapper.selectById(userId);
        PasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

        String databasePassword = user.getPassword();
        if(!passwordEncoder.matches(oldPassword,databasePassword)){
            throw new DiagnoseException("原密码错误");
        }

        String encodeNewPassword = passwordEncoder.encode(newPassword);
        user.setPassword(encodeNewPassword);
        userMapper.updateById(user);
    }

    private void validatePassword(String password) {
        // 正则表达式：密码长度至少为8位，且包含至少一个字母和一个数字
        String regex = "^(?=.*[A-Za-z])(?=.*\\d)[A-Za-z\\d]{8,}$";
        if(!Pattern.matches(regex, password)){
            throw new DiagnoseException("请输入八位以上密码，包含数字和字母");
        }
    }

}


