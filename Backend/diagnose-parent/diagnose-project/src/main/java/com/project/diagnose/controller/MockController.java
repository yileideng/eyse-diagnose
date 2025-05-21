package com.project.diagnose.controller;

import com.project.diagnose.aop.LogAnnotation;
import com.project.diagnose.client.MLClient;
import com.project.diagnose.dto.response.DiagnoseResponse;
import lombok.extern.slf4j.Slf4j;
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
    @Autowired
    private MLClient mlClient;

    @PostMapping("/send")
    public DiagnoseResponse mock(){
        DiagnoseResponse diagnoseResponse = null;
        try {
            File file = new File("C:Users/18101/Music/超级玛丽结束_爱给网_aigei_com.wav");
            List<File> fileList=new ArrayList<>();
            fileList.add(file);
            diagnoseResponse = mlClient.requestForBulkDiagnose(fileList);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return diagnoseResponse;
    }

    @PostMapping("/file-bulk")
    //@PreAuthorize("hasAuthority('upload')")
    @LogAnnotation(module = "FileController",operator = "用MinIO上传用户头像")
    public String mock(@RequestParam(value = "file", required = false) MultipartFile[] files) {

        return "{\n" +
                "  \"prediction_results\": {\n" +
                "    \"patient_c7b519d0\": {\n" +
                "      \"diseases\": {\n" +
                "        \"AMD\": {\n" +
                "          \"probability\": 0.00019706117745954543,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"其他疾病/异常\": {\n" +
                "          \"probability\": 0.3022720515727997,\n" +
                "          \"status\": 1\n" +
                "        },\n" +
                "        \"正常\": {\n" +
                "          \"probability\": 0.0006025025504641235,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"白内障\": {\n" +
                "          \"probability\": 1.0,\n" +
                "          \"status\": 1\n" +
                "        },\n" +
                "        \"糖尿病\": {\n" +
                "          \"probability\": 0.00016099114145617932,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"近视\": {\n" +
                "          \"probability\": 8.336865721503273e-05,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"青光眼\": {\n" +
                "          \"probability\": 0.00037023425102233887,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"高血压\": {\n" +
                "          \"probability\": 0.00020455196499824524,\n" +
                "          \"status\": 0\n" +
                "        }\n" +
                "      }\n" +
                "    },\n" +
                "    \"patient_c746519d2\": {\n" +
                "      \"diseases\": {\n" +
                "        \"AMD\": {\n" +
                "          \"probability\": 8.875525963958353e-06,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"其他疾病/异常\": {\n" +
                "          \"probability\": 0.9908466935157776,\n" +
                "          \"status\": 1\n" +
                "        },\n" +
                "        \"正常\": {\n" +
                "          \"probability\": 0.00014771672431379557,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"白内障\": {\n" +
                "          \"probability\": 0.3483974039554596,\n" +
                "          \"status\": 1\n" +
                "        },\n" +
                "        \"糖尿病\": {\n" +
                "          \"probability\": 0.9999253749847412,\n" +
                "          \"status\": 1\n" +
                "        },\n" +
                "        \"近视\": {\n" +
                "          \"probability\": 4.61619820271153e-05,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"青光眼\": {\n" +
                "          \"probability\": 7.147400174289942e-05,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"高血压\": {\n" +
                "          \"probability\": 0.0001380705798510462,\n" +
                "          \"status\": 0\n" +
                "        }\n" +
                "      }\n" +
                "    }\n" +
                "  }\n" +
                "}";
    }

    @PostMapping("/file-personal")
    //@PreAuthorize("hasAuthority('upload')")
    @LogAnnotation(module = "FileController",operator = "个人生成诊断报告")
    public String mockPerson(@RequestParam(value = "images", required = false) MultipartFile[] files) {

        return "{\n" +
                "  \"prediction_results\": {\n" +
                "    \"patient_c746534h436\": {\n" +
                "      \"diseases\": {\n" +
                "        \"AMD\": {\n" +
                "          \"probability\": 0.00019706117745954543,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"其他疾病/异常\": {\n" +
                "          \"probability\": 0.3022720515727997,\n" +
                "          \"status\": 1\n" +
                "        },\n" +
                "        \"正常\": {\n" +
                "          \"probability\": 0.0006025025504641235,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"白内障\": {\n" +
                "          \"probability\": 1.0,\n" +
                "          \"status\": 1\n" +
                "        },\n" +
                "        \"糖尿病\": {\n" +
                "          \"probability\": 0.00016099114145617932,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"近视\": {\n" +
                "          \"probability\": 8.336865721503273e-05,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"青光眼\": {\n" +
                "          \"probability\": 0.00037023425102233887,\n" +
                "          \"status\": 0\n" +
                "        },\n" +
                "        \"高血压\": {\n" +
                "          \"probability\": 0.00020455196499824524,\n" +
                "          \"status\": 0\n" +
                "        }\n" +
                "      }\n" +
                "    }  \n" +
                "  }\n" +
                "}";
    }
}
