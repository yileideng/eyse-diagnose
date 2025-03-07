package com.project.diagnose.exception;

import com.project.diagnose.dto.vo.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.AuthenticationException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

@Slf4j
@ControllerAdvice
public class GlobalExceptionHandler {
    // 自定义异常
    @ExceptionHandler(DiagnoseException.class)
    public ResponseEntity<Result> handleBlogException(DiagnoseException e) {
        log.error("捕获到BlogException: {}", e.getMessage());
        return ResponseEntity.status(e.getHttpStatus())
                .body(Result.error(e.getHttpStatus().value(), e.getMessage()));
    }
    
    // SpringSecurity认证不通过
    @ExceptionHandler(AuthenticationException.class)
    public ResponseEntity<Result> handleAuthenticationException(AuthenticationException e) {
        log.error("捕获到AuthenticationException: {}", e.getMessage());
        return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                .body(Result.error(HttpStatus.UNAUTHORIZED.value(), ErrorMessage.ACCOUNT_PWD_WRONG.getMsg()));
    }
    // 参数异常
    // SpringSecurity认证不通过
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Result> handleIllegalArgumentException(IllegalArgumentException e) {
        log.error("捕获到IllegalArgumentException: {}", e.getMessage());
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body(Result.error(HttpStatus.BAD_REQUEST.value(), e.getMessage()));
    }

/*    // 其它异常归为系统出错SYSTEM_ERROR
    @ExceptionHandler(Exception.class)
    public ResponseEntity<Result> handleException(Exception e) {
        log.error("捕获到Exception: {}", e.getMessage());
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(Result.error(HttpStatus.INTERNAL_SERVER_ERROR.value(), ErrorMessage.SYSTEM_ERROR.getMsg() + ": " + e.getMessage()));
    }*/
}
