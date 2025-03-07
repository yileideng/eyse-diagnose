package com.project.diagnose.filter;


import com.project.diagnose.pojo.LoginUser;
import com.project.diagnose.utils.JWTUtils;
import com.project.diagnose.utils.RedisUtils;
import io.jsonwebtoken.Claims;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Objects;

@Component
public class JwtAuthenticationTokenFilter extends OncePerRequestFilter {
    @Autowired
    private RedisUtils redisUtils;
    @Override
    protected void doFilterInternal(HttpServletRequest httpServletRequest, HttpServletResponse httpServletResponse, FilterChain filterChain) throws ServletException, IOException {
        String token=httpServletRequest.getHeader("Authorization");
        // 如果没有token：可能是登录请求，直接放行（后面的过滤器还会处理）
        if(!StringUtils.hasText(token)){
            // 认证阶段直接放行
            filterChain.doFilter(httpServletRequest,httpServletResponse);
            return;
        }

        try {
            // 检查token合法性
            Claims claims= (Claims) JWTUtils.checkToken(token);
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("token非法");
        }

        LoginUser loginUser = redisUtils.getLoginUserInRedis(token);
        if(Objects.isNull(loginUser)){
            throw new RuntimeException("用户未登录,或登录已过期");
        }

        // 传入三个参数会把"已认证"设置成true:底层是super.setAuthenticated(true);
        UsernamePasswordAuthenticationToken usernamePasswordAuthenticationToken=new UsernamePasswordAuthenticationToken(loginUser,null,loginUser.getAuthorities());
        // 把用户信息loginUser存入SecurityContextHolder
        SecurityContextHolder.getContext().setAuthentication(usernamePasswordAuthenticationToken);

        // 最后放行
        filterChain.doFilter(httpServletRequest,httpServletResponse);
    }
}
