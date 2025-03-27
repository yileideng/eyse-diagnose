package com.project.diagnose.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

import java.util.Map;

public class DiagnoseResponse {

    @JsonProperty("prediction_results")
    private Map<String, PredictionResult> predictionResults;
    private int predictionResultsSize;

    // 默认构造函数，Jackson需要这个来反序列化
    public DiagnoseResponse() {
    }

    // 构造函数（如果需要）
    public DiagnoseResponse(Map<String, PredictionResult> predictionResults) {
        this.predictionResults = predictionResults;
        this.predictionResultsSize = predictionResults.size();
    }

    // Getter和Setter
    public Map<String, PredictionResult> getPredictionResults() {
        return predictionResults;
    }

    public void setPredictionResults(Map<String, PredictionResult> predictionResults) {
        this.predictionResults = predictionResults;
    }

    public void setPredictionResultsSize(int size) {
        this.predictionResultsSize = size;
    }

    public int getPredictionResultsSize() {
        return predictionResultsSize;
    }

    // 内部类：表示每个X_right的预测结果
    public static class PredictionResult {
        private Map<String, DiseaseInfo> diseases;

        public PredictionResult() {
        }

        public PredictionResult(Map<String, DiseaseInfo> diseases) {
            this.diseases = diseases;
        }

        public Map<String, DiseaseInfo> getDiseases() {
            return diseases;
        }

        public void setDiseases(Map<String, DiseaseInfo> diseases) {
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
