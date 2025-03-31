package com.project.diagnose.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public class PersonalDiagnoseResponse {
    @JsonProperty("prediction_results")
    private Map<String, BulkDiagnoseResponse.PredictionResult> predictionResults;

    // 默认构造函数，Jackson需要这个来反序列化
    public PersonalDiagnoseResponse() {
    }

    // 构造函数（如果需要）
    public PersonalDiagnoseResponse(Map<String, BulkDiagnoseResponse.PredictionResult> predictionResults) {
        this.predictionResults = predictionResults;
    }

    // Getter和Setter
    public Map<String, BulkDiagnoseResponse.PredictionResult> getPredictionResults() {
        return predictionResults;
    }

    public void setPredictionResults(Map<String, BulkDiagnoseResponse.PredictionResult> predictionResults) {
        this.predictionResults = predictionResults;
    }

    // 内部类：表示每个X_right的预测结果
    public static class PredictionResult {
        private Map<String, BulkDiagnoseResponse.DiseaseInfo> diseases;

        public PredictionResult() {
        }

        public PredictionResult(Map<String, BulkDiagnoseResponse.DiseaseInfo> diseases) {
            this.diseases = diseases;
        }

        public Map<String, BulkDiagnoseResponse.DiseaseInfo> getDiseases() {
            return diseases;
        }

        public void setDiseases(Map<String, BulkDiagnoseResponse.DiseaseInfo> diseases) {
            this.diseases = diseases;
        }
    }

    // 内部类：表示每个疾病的预测信息
    public static class DiseaseInfo {
        private double probability;
        private int status;

        public DiseaseInfo() {
        }

        public DiseaseInfo(double probability, int status) {
            this.probability = probability;
            this.status = status;
        }

        public double getProbability() {
            return probability;
        }

        public void setProbability(double probability) {
            this.probability = probability;
        }

        public int getStatus() {
            return status;
        }

        public void setStatus(int status) {
            this.status = status;
        }
    }
}
