package com.project.diagnose.controller;

import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.client.MLClient;
import com.project.diagnose.dto.response.DiagnoseResponse;
import com.project.diagnose.dto.vo.Result;
import com.project.diagnose.service.AvatarImageService;
import com.project.diagnose.utils.FileUtils;
import com.project.diagnose.utils.RedisUtils;
import lombok.extern.slf4j.Slf4j;
import okhttp3.OkHttpClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@RestController
@RequestMapping("/mock")
public class MockController {
    @Value("${minio.bucket.avatar}")
    private String bucket;
    @Autowired
    private MLClient mlClient;

    @PostMapping("/send")
    public DiagnoseResponse mock(){
        DiagnoseResponse diagnoseResponse = null;
        try {
            File file = new File("C:Users/18101/Music/超级玛丽结束_爱给网_aigei_com.wav");
            List<File> fileList=new ArrayList<>();
            fileList.add(file);
            diagnoseResponse = mlClient.requestForDiagnose(fileList);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return diagnoseResponse;
    }

    @PostMapping("/files")
    //@PreAuthorize("hasAuthority('upload')")
    @LogAnnotation(module = "FileController",operator = "用MinIO上传用户头像")
    public String mock(@RequestParam(value = "fileList", required = false) MultipartFile[] files) {

        return "{\n" +
                "  \"prediction_results\": {\n" +
                "    \"diseases\": {\n" +
                "      \"正常\": {\n" +
                "        \"probability\": 0.9979015588760376,\n" +
                "        \"status\": 1\n" +
                "      },\n" +
                "      \"糖尿病\": {\n" +
                "        \"probability\": 0.00024866973399184644,\n" +
                "        \"status\": 0\n" +
                "      },\n" +
                "      \"青光眼\": {\n" +
                "        \"probability\": 0.3476810157299042,\n" +
                "        \"status\": 1\n" +
                "      },\n" +
                "      \"白内障\": {\n" +
                "        \"probability\": 0.08230943232774734,\n" +
                "        \"status\": 0\n" +
                "      },\n" +
                "      \"AMD\": {\n" +
                "        \"probability\": 1.1338810281813494e-06,\n" +
                "        \"status\": 0\n" +
                "      },\n" +
                "      \"高血压\": {\n" +
                "        \"probability\": 3.5314155866217334e-06,\n" +
                "        \"status\": 0\n" +
                "      },\n" +
                "      \"近视\": {\n" +
                "        \"probability\": 0.0008877236396074295,\n" +
                "        \"status\": 0\n" +
                "      },\n" +
                "      \"其他疾病/异常\": {\n" +
                "        \"probability\": 0.7849184274673462,\n" +
                "        \"status\": 1\n" +
                "      }\n" +
                "    }\n" +
                "  }\n" +
                "}";
    }
}
