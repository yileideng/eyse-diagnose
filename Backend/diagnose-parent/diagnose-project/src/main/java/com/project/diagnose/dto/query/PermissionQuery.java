package com.project.diagnose.dto.query;

import lombok.Data;

import java.util.List;

@Data
public class PermissionQuery {
    private Long id;
    private String path;
    private List<Long> roleIds;
    private String comment;
}
