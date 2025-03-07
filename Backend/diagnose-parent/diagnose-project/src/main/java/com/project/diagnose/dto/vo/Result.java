package com.project.diagnose.dto.vo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Result<T> {

    private Integer code;

    private String msg;

    private T data;


    public static <T> Result<T> success(T data){return new Result<>(200,"success",data);}
    public static Result<Boolean> success(){return new Result<>(200,"success",true);}

    public static Result<Boolean> error(int code, String msg){
        return new Result<>(code,msg,false);
    }
}

