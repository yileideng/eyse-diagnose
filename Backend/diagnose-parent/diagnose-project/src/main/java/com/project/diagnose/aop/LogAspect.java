package com.project.diagnose.aop;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.serializer.SerializerFeature;
import com.project.diagnose.dto.vo.Result;
import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;
import org.aspectj.lang.reflect.MethodSignature;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import javax.servlet.http.HttpServletRequest;
import java.lang.reflect.Method;

@Component
@Aspect
@Slf4j
public class LogAspect {
    @Value("${logging.aop.enabled}")
    private boolean logEnabled;

    @Value("${logging.aop.log-request}")
    private boolean logRequest;

    @Value("${logging.aop.log-response}")
    private boolean logResponse;

    @Pointcut("@annotation(com.project.diagnose.aop.LogAnnotation)")
    public void pt() {}

    @Around("pt()")
    public Object log(ProceedingJoinPoint joinPoint) throws Throwable {
        long beginTime = System.currentTimeMillis();
        if(logEnabled){
            try {
                beforeProceedLog(joinPoint);
            } catch (Exception e) {
                log.error("aop记录方法执行前的日志时出错: {}", e.getMessage());
            }

        }

        // 千万不要用try-catch捕获异常,否则全局异常处理器将无法捕获异常
        Object result = joinPoint.proceed();

        long time = System.currentTimeMillis() - beginTime;
        if (logEnabled) {
            // 记录日志不能影响程序的正常运行,捕获日志中可能会的抛出异常,
            try {
                afterProceedLog(joinPoint, time, result);
            }catch (Exception e){
                log.error("aop记录方法执行后的日志时出错: {}", e.getMessage());
            }
        }

        return result;
    }
    private void beforeProceedLog(ProceedingJoinPoint joinPoint) {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();
        LogAnnotation logAnnotation = method.getAnnotation(LogAnnotation.class);

        log.info("===================== log start ================================");
        log.info("Module: {}", logAnnotation.module());
        log.info("Operation: {}", logAnnotation.operator());

        String className = joinPoint.getTarget().getClass().getName();
        String methodName = signature.getName();
        log.info("Request Method: {}", className + "." + methodName + "()");

        if (logRequest) {
            // 获取请求对象
            ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            if (attributes != null) {
                HttpServletRequest request = attributes.getRequest();
                // 获取请求 IP
                log.info("Request IP: {}", request.getRemoteAddr());
            }

            Object[] args = joinPoint.getArgs();
            if (args != null && args.length > 0) {
                String params = JSON.toJSONString(args, SerializerFeature.IgnoreNonFieldGetter);
                log.info("Params: {}", params);
            } else {
                log.info("Params: No parameters");
            }
        }
    }
    private void afterProceedLog(ProceedingJoinPoint joinPoint, long time, Object result) {
        MethodSignature signature = (MethodSignature) joinPoint.getSignature();
        Method method = signature.getMethod();

        if (logResponse) {
            // 如果是Controller层的方法
            if (method.getDeclaringClass().isAnnotationPresent(RestController.class) && method.getReturnType() == Result.class) {
                Result restResult = (Result) result;
                log.info("HTTP Code: {}", restResult.getCode());
                log.info("Response Message: {}", restResult.getMsg());
            }
            /*else {
                log.info("Result: {}", JSON.toJSONString(result));
            }*/
        }

        log.info("Execution time: {}ms", time);
        log.info("==============================================================");
    }
}