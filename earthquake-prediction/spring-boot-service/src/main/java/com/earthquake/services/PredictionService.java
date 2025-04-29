package com.earthquake.services;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class PredictionService {

    private final WebClient pythonServiceClient;

    /**
     * Get model configuration from Python service
     */
    public Mono<Map<String, Object>> getModelConfig() {
        return pythonServiceClient.get()
                .uri("/config")
                .retrieve()
                .bodyToMono(Map.class);
    }

    /**
     * Send prediction request to Python service
     */
    public Mono<Map<String, Object>> predict(Map<String, Object> request) {
        return pythonServiceClient.post()
                .uri("/predict")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(Map.class);
    }
    
    /**
     * Get available models from Python service
     */
    public Mono<Map<String, Object>> getAvailableModels() {
        return pythonServiceClient.get()
                .uri("/models")
                .retrieve()
                .bodyToMono(Map.class);
    }
    
    /**
     * Activate a different model
     */
    public Mono<Map<String, Object>> activateModel(Map<String, Object> request) {
        return pythonServiceClient.post()
                .uri("/models/activate")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(Map.class);
    }
}