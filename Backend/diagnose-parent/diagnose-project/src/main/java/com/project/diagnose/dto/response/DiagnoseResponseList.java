package com.project.diagnose.dto.response;

import lombok.Data;

import java.util.List;

@Data
public class DiagnoseResponseList {

    private String diagnoseMode;
    private List<DiagnoseResponse.PredictionResult> predictionResultsList;


    public DiagnoseResponseList() {
    }

    public DiagnoseResponseList(List<DiagnoseResponse.PredictionResult> predictionResultsList, String diagnoseMode) {
        this.predictionResultsList = predictionResultsList;
        this.diagnoseMode = diagnoseMode;
    }

    public List<DiagnoseResponse.PredictionResult> getPredictionResultsList() {
        return predictionResultsList;
    }

    public void setPredictionResultsList(List<DiagnoseResponse.PredictionResult> predictionResultsList) {
        this.predictionResultsList = predictionResultsList;
    }
}
