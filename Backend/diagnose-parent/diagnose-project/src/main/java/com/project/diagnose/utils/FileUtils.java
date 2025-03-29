package com.project.diagnose.utils;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.HashMap;

public class FileUtils {
    @NoArgsConstructor
    @AllArgsConstructor
    @Getter
    public enum Category {
        CATEGORY_IMAGE("image"),
        CATEGORY_ZIP("zip");

        private String category;
    }

    private static final HashMap<String, String> MIME_TYPES = new HashMap<>();
    private static final HashMap<String, Category> MIME_TYPE_FUNCTIONS = new HashMap<>();

    static {
        // 添加更多的 MIME 类型映射
        MIME_TYPES.put(".txt", "text/plain");
        MIME_TYPES.put(".html", "text/html");
        MIME_TYPES.put(".css", "text/css");
        MIME_TYPES.put(".js", "application/javascript");
        MIME_TYPES.put(".json", "application/json");
        MIME_TYPES.put(".jpg", "image/jpeg");
        MIME_TYPES.put(".jpeg", "image/jpeg"); // 添加 jpeg
        MIME_TYPES.put(".png", "image/png");
        MIME_TYPES.put(".gif", "image/gif");
        MIME_TYPES.put(".pdf", "application/pdf");
        MIME_TYPES.put(".mp3", "audio/mpeg");
        MIME_TYPES.put(".mp4", "video/mp4");
        MIME_TYPES.put(".doc", "application/msword");
        MIME_TYPES.put(".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"); // 添加 docx
        MIME_TYPES.put(".xls", "application/vnd.ms-excel");
        MIME_TYPES.put(".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"); // 添加 xlsx
        MIME_TYPES.put(".xml", "text/xml");
        MIME_TYPES.put(".zip", "application/zip");
        MIME_TYPES.put(".csv", "text/csv");
        MIME_TYPES.put(".svg", "image/svg+xml");
        MIME_TYPES.put(".ppt", "application/vnd.ms-powerpoint");
        MIME_TYPES.put(".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"); // 添加 pptx

        // 头像
        MIME_TYPE_FUNCTIONS.put("image/jpeg", Category.CATEGORY_IMAGE);
        MIME_TYPE_FUNCTIONS.put("image/png", Category.CATEGORY_IMAGE);
        MIME_TYPE_FUNCTIONS.put("image/gif", Category.CATEGORY_IMAGE);
        MIME_TYPE_FUNCTIONS.put("application/zip", Category.CATEGORY_ZIP);


    }

    public static String getMimeType(String fileName) {
        // 提取文件扩展名，并转换为小写（不区分大小写）
        String extension = "";
        if (fileName != null && fileName.contains(".")) {
            int index = fileName.lastIndexOf(".");
            extension = fileName.substring(index).toLowerCase();
        }

        // 返回对应的 MIME 类型，如果不存在返回默认值
        String mimeType = MIME_TYPES.get(extension);
        if (mimeType == null) {
            // 如果没有找到 MIME 类型，记录日志并返回默认值
            System.out.println("未找到文件扩展名 " + extension + " 的 MIME 类型，返回默认值: application/octet-stream");
            return "application/octet-stream";
        }
        return mimeType;
    }

    public static Category getMimeTypeFunction(String mimeType) {
        // 返回对应的 MIME 类型，如果不存在返回默认值
        return MIME_TYPE_FUNCTIONS.getOrDefault(mimeType, null);
    }

    public static Boolean checkFileCategory(String fileName, Category requiredCategory) {
        String mimeType = FileUtils.getMimeType(fileName);
        Category fileCategory = FileUtils.getMimeTypeFunction(mimeType);
        return fileCategory == requiredCategory;
    }
}