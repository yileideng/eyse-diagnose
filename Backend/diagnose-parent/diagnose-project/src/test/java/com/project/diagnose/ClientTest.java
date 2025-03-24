package com.project.diagnose;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.project.diagnose.dto.response.DiagnoseResponse;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class ClientTest {
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final String response = "{\n" +
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

    @Test
    void jsonResponse() throws Exception {
        DiagnoseResponse diagnoseResponse = objectMapper.readValue(response, DiagnoseResponse.class);
        System.out.println(diagnoseResponse.toString());
    }
}