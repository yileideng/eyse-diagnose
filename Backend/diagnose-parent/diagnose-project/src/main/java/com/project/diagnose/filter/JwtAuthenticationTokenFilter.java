package com.project.diagnose.filter;


import com.alibaba.fastjson.JSON;
import com.project.diagnose.dto.vo.Result;
import com.project.diagnose.pojo.LoginUser;
import com.project.diagnose.utils.JWTUtils;
import com.project.diagnose.utils.RedisUtils;
import io.jsonwebtoken.Claims;
import lombok.extern.slf4j.Slf4j;
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

@Slf4j
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
            // 捕获JWT校验失败的异常，返回401 Unauthorized状态码
            httpServletResponse.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            httpServletResponse.setContentType("application/json;charset=UTF-8");
            httpServletResponse.getWriter().write(JSON.toJSONString(Result.error(HttpServletResponse.SC_UNAUTHORIZED, "Token无效或已过期")));
            log.info("Token无效或已过期");
            return;
        }

        LoginUser loginUser = redisUtils.getLoginUserInRedis(token);
        if(Objects.isNull(loginUser)){
            // 用户未登录或登录已过期，返回401 Unauthorized状态码
            httpServletResponse.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
            httpServletResponse.setContentType("application/json;charset=UTF-8");
            httpServletResponse.getWriter().write(JSON.toJSONString(Result.error(HttpServletResponse.SC_UNAUTHORIZED, "用户未登录或登录已过期")));
            log.info("用户未登录或登录已过期");
            return;
        }

        // 传入三个参数会把"已认证"设置成true:底层是super.setAuthenticated(true);
        UsernamePasswordAuthenticationToken usernamePasswordAuthenticationToken=new UsernamePasswordAuthenticationToken(loginUser,null,loginUser.getAuthorities());
        // 把用户信息loginUser存入SecurityContextHolder
        SecurityContextHolder.getContext().setAuthentication(usernamePasswordAuthenticationToken);

        // 最后放行
        filterChain.doFilter(httpServletRequest,httpServletResponse);
    }
}
