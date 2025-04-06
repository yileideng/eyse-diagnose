package com.project.diagnose.controller;

import com.project.diagnose.dto.vo.Result;
import com.project.diagnose.service.DeepSeekService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/deepseek")
public class DeepSeekController {
    @Autowired
    private DeepSeekService deepSeekService;
    @PostMapping("/interact")
    public Result<String> interact(@RequestParam String prompt) {
        String interact = deepSeekService.interact(prompt);
        return Result.success(interact);
    }
}
