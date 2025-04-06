

package com.project.diagnose.service.Impl;

import com.baomidou.mybatisplus.core.toolkit.StringUtils;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.mysql.cj.x.protobuf.MysqlxDatatypes;
import com.project.diagnose.dto.query.UserQuery;
import com.project.diagnose.dto.vo.UserVo;
import com.project.diagnose.exception.DiagnoseException;
import com.project.diagnose.mapper.UserMapper;
import com.project.diagnose.pojo.LoginUser;
import com.project.diagnose.pojo.User;
import com.project.diagnose.dto.query.LoginQuery;
import com.project.diagnose.service.LoginService;
import com.project.diagnose.service.PermissionService;
import com.project.diagnose.service.UserService;
import com.project.diagnose.utils.JWTUtils;
import com.project.diagnose.exception.ErrorMessage;
import com.project.diagnose.dto.vo.LoginVo;
import com.project.diagnose.utils.MailUtils;
import com.project.diagnose.utils.RedisUtils;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.regex.Pattern;

@Slf4j
@Service
@Transactional
public class LoginServiceImpl extends ServiceImpl<UserMapper, User> implements LoginService {

    @Autowired
    private PasswordEncoder passwordEncoder;
    @Autowired
    private AuthenticationManager authenticationManager;
    @Autowired
    private UserService userService;
    @Autowired
    private PermissionService permissionService;
    @Autowired
    private MailUtils mailUtils;
    @Autowired
    private RedisUtils redisUtils;
    @Autowired
    private UserMapper userMapper;


   /* //password的加密盐
    public static final String salt = "lingrui@#!%&";*/

    // 登录
    @Override
    public LoginVo passwordLogin(LoginQuery loginQuery) {

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
    @Override
    public LoginVo emailLogin(LoginQuery loginQuery) {
        String email = loginQuery.getEmail();
        String code = loginQuery.getCode();

        // 校验验证码
        checkCode(email, code);

        // 查询用户是否存在
        User userByMail = userService.findUserByMail(email);

        if (Objects.isNull(userByMail)) {
            // 如果用户不存在，创建一个新用户
            userByMail = new User();

            // 设置随机用户名
            String randomUsername = generateRandomUsername(email);
            userByMail.setUsername(randomUsername);
            userByMail.setEmail(email);
            // 设置随机密码（更安全的做法）
            String randomPassword = generateRandomPassword();
            userByMail.setPassword(passwordEncoder.encode(randomPassword));
            // 默认角色为普通用户
            userByMail.setRoleId(0L);
            userByMail.setUpdateTime(LocalDateTime.now());

            // 对新用户发送欢迎邮件
            try {
                String subject = "欢迎新用户注册";
                StringBuilder content = new StringBuilder();
                content.append("<p>尊敬的用户，</p>");
                content.append("<p>欢迎注册我们的平台！以下是您的账户信息：</p>");
                content.append("<p><strong>用户名：</strong>").append(randomUsername).append("</p>");
                content.append("<p><strong>邮箱：</strong>").append(email).append("</p>");
                content.append("<p><strong>初始密码：</strong>").append(randomPassword).append("</p>");
                content.append("<p>为了保障您的账户安全，请在首次登录后及时修改密码。</p>");
                content.append("<p>如果您在使用过程中遇到任何问题，可以随时联系我们的客服团队，我们将竭诚为您服务。</p>");
                content.append("<p>感谢您选择我们的服务，祝您使用愉快！</p>");
                content.append("<p>如果您不是本人操作，请忽略此邮件。</p>");

                mailUtils.sendSimpleMail(email, subject, content.toString());
            } catch (Exception e) {
                throw new DiagnoseException("邮件发送失败，请检查邮箱是否正确", HttpStatus.BAD_REQUEST);
            }

            // 保存用户到数据库
            userService.save(userByMail);
        }

        // 删除Redis中的验证码
        redisUtils.deleteCode(email);

        // 获取用户权限
        List<String> permissionList = permissionService.findPermissionPathByUserId(userByMail.getId());

        // 登录逻辑（生成 JWT 令牌并返回）
        LoginUser loginUser = new LoginUser(userByMail, permissionList);
        String token = JWTUtils.createToken(loginUser.getUser().getId());
        redisUtils.setLoginUserInRedis(token, loginUser);

        LoginVo loginVo = new LoginVo(loginUser);
        loginVo.setToken(token);

        return loginVo;
    }
    // 校验验证码
    private void checkCode(String email, String code) {
        if (StringUtils.isBlank(code)) {
            throw new DiagnoseException("请输入验证码", HttpStatus.BAD_REQUEST);
        }

        String redisCode = redisUtils.getCode(email);
        if (StringUtils.isBlank(redisCode)) {
            throw new DiagnoseException("验证码已过期", HttpStatus.UNAUTHORIZED);
        }

        if (!code.equals(redisCode)) {
            throw new DiagnoseException("验证码错误", HttpStatus.BAD_REQUEST);
        }
    }
    // 生成随机且唯一用户名
    private String generateRandomUsername(String email) {
        String emailPrefix = email.split("@")[0];
        int randomSuffix = new Random().nextInt(9000) + 1000; // 4位随机数字
        String randomUsername = emailPrefix + "_" + randomSuffix;

        // 确保用户名唯一
        while (userService.findUserByUsername(randomUsername) != null) {
            randomSuffix = new Random().nextInt(9000) + 1000;
            randomUsername = emailPrefix + "_" + randomSuffix;
        }

        return randomUsername;
    }
    // 生成随机密码
    private String generateRandomPassword() {
        // 生成随机密码（例如8位，包含字母和数字）
        String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        StringBuilder password = new StringBuilder();
        Random random = new Random();
        for (int i = 0; i < 8; i++) {
            password.append(chars.charAt(random.nextInt(chars.length())));
        }
        return password.toString();
    }
    //退出登录
    @Override
    public void logout(String token) {
        redisUtils.deleteLoginUserInRedis(token);
    }

    @Override
    public void generateMail(String mail) {
        String code = randomCode();
        try {
            String subject = "您的邮箱验证码";
            StringBuilder content = new StringBuilder();
            content.append("<p>尊敬的用户，</p>");
            content.append("<p>您正在使用邮箱验证码功能，以下是您的验证码：</p>");
            content.append("<p><strong>").append(code).append("</strong></p>");
            content.append("<p>验证码有效期为1分钟，请尽快使用。</p>");
            content.append("<p>如果您没有请求邮箱验证码，可能是有人误操作。您可以忽略此邮件，或者联系我们的客服团队。</p>");
            content.append("<p>感谢您使用我们的服务！</p>");
            content.append("<p>此致</p>");
            log.info("邮件内容构建完成");

            mailUtils.sendSimpleMail(mail, subject, content.toString());
        } catch (Exception e) {
            log.info("发送验证码失败： {}", e.getMessage(),e);
            throw new DiagnoseException("发送验证码失败，请检查邮箱格式", HttpStatus.BAD_REQUEST);
        }
        // 缓存邮件验证码
        redisUtils.setCode(mail, code);
        return;
    }
    private String randomCode(){
        int code = new Random().nextInt(900000) + 100000; // 范围是 100000 到 999999
        return String.valueOf(code);
    }

    //注册
    @Override
    public void register(LoginQuery loginQuery) {
        String password = loginQuery.getPassword();
        String username = loginQuery.getUsername();
        String phoneNumber = loginQuery.getPhoneNumber();
        String email = loginQuery.getEmail();
        String code = loginQuery.getCode();


        // 校验用户名和密码是否为空
        if (StringUtils.isBlank(password) || StringUtils.isBlank(username)|| StringUtils.isBlank(email) || StringUtils.isBlank(code)) {
            throw new DiagnoseException("请填写完整注册数据", HttpStatus.BAD_REQUEST);
        }

        // 校验密码复杂度
        if (!isValidPassword(password)) {
            throw new DiagnoseException("请输入八位以上密码，包含数字和字母", HttpStatus.BAD_REQUEST);
        }
        // 校验手机号格式
        if (!StringUtils.isBlank(phoneNumber) && !isValidPhoneNumber(phoneNumber)) {
            throw new DiagnoseException("请输入有效的手机号", HttpStatus.BAD_REQUEST);
        }
        // 校验邮箱格式
        if (!isValidEmail(email)) {
            throw new DiagnoseException("请输入有效的邮箱", HttpStatus.BAD_REQUEST);
        }

        // 检查用户名是否已存在
        User user = userService.findUserByUsername(username);
        if (user != null) {
            throw new DiagnoseException(ErrorMessage.ACCOUNT_EXIST, HttpStatus.CONFLICT);
        }
        // 检查邮箱是否已经存在
        User userByMail = userService.findUserByMail(email);
        if (userByMail != null) {
            throw new DiagnoseException("该邮箱已被注册", HttpStatus.CONFLICT);
        }

        // 验证验证码
        String realCode = redisUtils.getCode(email);
        if(!code.equals(realCode)){
            throw new DiagnoseException("验证码错误", HttpStatus.BAD_REQUEST);
        }
        redisUtils.deleteCode(email);
        // 发送欢迎邮件
        try {
            String subject = "欢迎新用户注册";
            StringBuilder content = new StringBuilder();
            content.append("尊敬的用户 ").append(username).append("：<br><br>");
            content.append("欢迎注册我们的平台！以下是您的账户信息：<br><br>");
            content.append("用户名：").append(username).append("<br>");
            content.append("邮箱：").append(email).append("<br>");
            content.append("手机号：").append(phoneNumber).append("<br><br>");
            content.append("如果您在使用过程中遇到任何问题，可以随时联系我们的客服团队，我们将竭诚为您服务。<br><br>");
            content.append("祝您使用愉快！<br><br>");
            content.append("如果您不是本人操作，请忽略此邮件。");

            mailUtils.sendSimpleMail(email, subject, content.toString());
        } catch (Exception e) {
            throw new DiagnoseException("邮件发送失败，请检查邮箱是否正确", HttpStatus.BAD_REQUEST);
        }

        // 如果账号未被注册，则将用户信息添加到数据库
        user = new User();
        user.setUsername(username);
        user.setPassword(passwordEncoder.encode(password));
        user.setUpdateTime(LocalDateTime.now());
        user.setRoleId(0L);
        user.setEmail(email);
        user.setPhoneNumber(phoneNumber);
        user.setAvatarUrl(loginQuery.getAvatarUrl());
        this.userService.save(user);
    }
    private boolean isValidPassword(String password) {
        // 正则表达式：密码长度至少为8位，且包含至少一个字母和一个数字
        String regex = "^(?=.*[A-Za-z])(?=.*\\d)[A-Za-z\\d]{8,}$";
        return Pattern.matches(regex, password);
    }
    // 校验手机号格式
    private boolean isValidPhoneNumber(String phoneNumber) {
        // 正则表达式：中国大陆手机号（11位数字，以1开头）
        String regex = "^1[3-9]\\d{9}$";
        return Pattern.matches(regex, phoneNumber);
    }
    // 校验邮箱格式
    private boolean isValidEmail(String email) {
        // 正则表达式：基本邮箱格式校验
        String regex = "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$";
        return Pattern.matches(regex, email);
    }

    @Override
    public void forgetPassword(LoginQuery loginQuery) {
        String email = loginQuery.getEmail();
        String code = loginQuery.getCode();
        String newPassword = loginQuery.getPassword();

        if (email == null || code == null || newPassword == null) {
            throw new DiagnoseException("请输入完整表格数据");
        }

        String realCode = redisUtils.getCode(email);
        if (realCode == null) {
            throw new DiagnoseException("验证码已过期");
        }
        if (!code.equals(realCode)) {
            throw new DiagnoseException("验证码错误");
        }
        if(!isValidPassword(newPassword)){
            throw new DiagnoseException("请输入8位以上密码，数字+字母");
        }

        User user = userService.findUserByMail(email);
        if (user == null) {
            throw new DiagnoseException("用户不存在");
        }

        user.setPassword(passwordEncoder.encode(newPassword));
        userMapper.updateById(user);
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


