package com.project.diagnose.dto.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Data
public class DeepSeekRequestModel {

    private static final int maxLength = 40;

    /**
     * 所用DeepSeek模型
     */
    private String model;
    private List<Message> messages;

    public DeepSeekRequestModel(){
        this.messages = new ArrayList<>();
    }
    public DeepSeekRequestModel(String model, Message message){
        this.messages = new ArrayList<>();
        this.messages.add(message);
        this.model = model;

    }

    public void addMessage(Message message){
        messages.add(message);
        // 最多只能缓存maxLength条消息
        if (messages.size() > maxLength) {
            messages = messages.subList(messages.size() - maxLength, messages.size());
        }
    }

    /**
     * 消息体
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Message {
        private String role;
        private String content;
    }
}

