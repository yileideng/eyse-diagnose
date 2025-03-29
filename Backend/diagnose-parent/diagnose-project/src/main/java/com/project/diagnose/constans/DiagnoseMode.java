package com.project.diagnose.constans;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum DiagnoseMode {
    PERSONAL("personal"),
    BULK("bulk");

    private String value;


}
