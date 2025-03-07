package com.project.diagnose.dto.vo;

import com.project.diagnose.pojo.LoginUser;
import com.project.diagnose.pojo.User;
import lombok.Data;
import lombok.NoArgsConstructor;

//返回给前端的数据
@Data
@NoArgsConstructor
public class UserVo extends PageVo{
    private String id;

    private String username;

    private String email;

    private String phoneNumber;

    private String updateTime;

    private String avatarUrl;

    private String roleId;

    public UserVo(LoginUser loginUser) {
        User user = loginUser.getUser();
        this.id= String.valueOf(user.getId());
        this.username= user.getUsername();
        this.email= user.getEmail();
        this.phoneNumber= user.getPhoneNumber();
        this.updateTime= user.getUpdateTime().toString();
        this.avatarUrl= user.getAvatarUrl();
        this.roleId= String.valueOf(user.getRoleId());
    }

    public UserVo(User user) {
        this.id = String.valueOf(user.getId());
        this.username = user.getUsername();
        this.email = user.getEmail();
        this.phoneNumber = user.getPhoneNumber();
        this.updateTime= user.getUpdateTime().toString();
        this.avatarUrl= user.getAvatarUrl();
        this.roleId= String.valueOf(user.getRoleId());
    }
}
