package com.project.diagnose.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.Map;

public class DiagnoseResponse {

    @JsonProperty("prediction_results")
    private PredictionResults predictionResults;

    // Getter and Setter
    public PredictionResults getPredictionResults() {
        return predictionResults;
    }

    public void setPredictionResults(PredictionResults predictionResults) {
        this.predictionResults = predictionResults;
    }

    public static class PredictionResults {
        @JsonProperty("diseases")
        private Map<String, Disease> diseases;

        // Getter and Setter
        public Map<String, Disease> getDiseases() {
            return diseases;
        }

        public void setDiseases(Map<String, Disease> diseases) {
            this.diseases = diseases;
        }
    }

    public static class Disease {
        @JsonProperty("probability")
        private double probability;

        @JsonProperty("status")
        private int status;

        // Getter and Setter
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

        @Override
        public String toString() {
            return "Disease{" +
                    "probability=" + probability +
                    ", status=" + status +
                    '}';
        }
    }

    @Override
    public String toString() {
        return "DiagnoseResponse{" +
                "predictionResults=" + predictionResults +
                '}';
    }
}
