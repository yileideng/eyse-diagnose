package com.project.diagnose.dto.response;

import lombok.Data;

import java.util.List;

@Data
public class BulkDiagnoseResponseList {

    private String diagnoseMode;
    private List<BulkDiagnoseResponse.PredictionResult> predictionResultsList;


    public BulkDiagnoseResponseList() {
    }

    public BulkDiagnoseResponseList(List<BulkDiagnoseResponse.PredictionResult> predictionResultsList, String diagnoseMode) {
        this.predictionResultsList = predictionResultsList;
        this.diagnoseMode = diagnoseMode;
    }

    public List<BulkDiagnoseResponse.PredictionResult> getPredictionResultsList() {
        return predictionResultsList;
    }

    public void setPredictionResultsList(List<BulkDiagnoseResponse.PredictionResult> predictionResultsList) {
        this.predictionResultsList = predictionResultsList;
    }
}
