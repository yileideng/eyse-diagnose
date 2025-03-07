package com.project.diagnose.exception;

import org.springframework.http.HttpStatus;

public class DiagnoseException extends RuntimeException {
    private HttpStatus httpStatus;


    public DiagnoseException(String message) {
        super(message);
        httpStatus = HttpStatus.INTERNAL_SERVER_ERROR;
    }
    public DiagnoseException(ErrorMessage errorMessage) {
        super(errorMessage.getMsg());
        httpStatus = HttpStatus.INTERNAL_SERVER_ERROR;
    }
    public DiagnoseException(String message, HttpStatus httpStatus) {
        super(message);
        this.httpStatus = httpStatus;
    }
    public DiagnoseException(ErrorMessage errorMessage, HttpStatus httpStatus) {
        super(errorMessage.getMsg());
        this.httpStatus = httpStatus;
    }

    public HttpStatus getHttpStatus() {
        return httpStatus;
    }
}
