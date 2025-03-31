

package com.project.diagnose.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.project.diagnose.pojo.User;
import com.project.diagnose.dto.query.UserQuery;
import com.project.diagnose.dto.vo.PageVo;
import com.project.diagnose.dto.vo.UserVo;

public interface UserService extends IService<User> {
    //用户分页查询
    PageVo<UserVo> getPage(UserQuery query);

    //根据账号密码查询token
    User findUser(String account, String password);

    //根据账号查询用户
    User findUserByUsername(String username);

    //根据id查询返回给前端的用户信息
    UserVo getUserVoById(Long id);

    void updateUser(Long userId, UserQuery user);

    User findUserByMail(String email);

    void updatePassword(Long userId, UserQuery userQuery);


}


