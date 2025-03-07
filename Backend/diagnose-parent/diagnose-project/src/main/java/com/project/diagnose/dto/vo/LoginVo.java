package com.project.diagnose.dto.vo;

import com.project.diagnose.pojo.LoginUser;
import com.project.diagnose.pojo.User;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
public class LoginVo {

    private String token;

    private String userId;

    private String username;

    private String roleId;

    private String avatarUrl;

    public LoginVo(LoginUser loginUser){
        User user = loginUser.getUser();
        this.userId= String.valueOf(user.getId());
        this.username=user.getUsername();
        this.roleId= String.valueOf(user.getRoleId());
        this.avatarUrl=user.getAvatarUrl();
    }

}
