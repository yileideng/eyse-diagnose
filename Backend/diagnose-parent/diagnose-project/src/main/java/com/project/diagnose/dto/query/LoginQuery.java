package com.project.diagnose.dto.query;

import lombok.Data;

@Data
public class LoginQuery {

    private String username;

    private String password;

    private  String email;

    private String phoneNumber;

    private String avatarUrl;

    private String code;
}
