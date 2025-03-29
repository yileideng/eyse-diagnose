package com.project.diagnose.utils;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import javax.mail.MessagingException;
import javax.mail.internet.MimeMessage;

@Component
public class MailUtils {
    @Value("${spring.mail.username}")
    private String fromMail;

    @Autowired
    private JavaMailSender mailSender;

    public void sendSimpleMail(String toMail, String subject, String text) throws MessagingException {
        MimeMessage message = mailSender.createMimeMessage();
        MimeMessageHelper helper = new MimeMessageHelper(message);
        helper.setFrom(fromMail);
        helper.setTo(toMail);
        // 邮件主题
        helper.setSubject(subject);
        // 邮件正文
        helper.setText(text);
        mailSender.send(message);
    }

    public void sendAttachmentMail(String toMail, String subject, String text, MultipartFile attachment) throws Exception {
        // 检查附件是否为空
        if (attachment == null || attachment.isEmpty()) {
            throw new IllegalArgumentException("附件不能为空");
        }

        MimeMessage message = mailSender.createMimeMessage();
        try {
            MimeMessageHelper helper = new MimeMessageHelper(message, true); // true表示支持多部件
            helper.setFrom(fromMail);
            helper.setTo(toMail);
            helper.setSubject(subject);
            helper.setText(text);

            // 获取附件的文件名
            String fileName = attachment.getOriginalFilename();
            if (fileName == null || fileName.isEmpty()) {
                throw new IllegalArgumentException("附件文件名不能为空");
            }

            // 添加附件
            helper.addAttachment(fileName, attachment);

            // 发送邮件
            mailSender.send(message);
        } catch (Exception e) {
            // 记录异常日志
            System.err.println("发送带附件的邮件时发生异常: " + e.getMessage());
            throw e; // 抛出异常供调用者处理
        }
    }

    public void sendHtmlMail(String toMail, String subject, String html) throws MessagingException {
        MimeMessage message = mailSender.createMimeMessage();
        MimeMessageHelper helper = new MimeMessageHelper(message, true);
        helper.setFrom(fromMail);
        helper.setTo(toMail);
        helper.setSubject(subject);
        helper.setText(html, true);
        mailSender.send(message);
    }
}

