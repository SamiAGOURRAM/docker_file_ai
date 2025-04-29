package com.earthquake.services;

import lombok.RequiredArgsConstructor;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class PredictionService {

    private final WebClient pythonServiceClient;

    // carry the full Map<String,Object> generic type through to runtime
    private static final ParameterizedTypeReference<Map<String, Object>> MAP_TYPE_REF =
        new ParameterizedTypeReference<Map<String, Object>>() {};

    /**
     * Get model configuration from Python service
     */
    public Mono<Map<String, Object>> getModelConfig() {
        return pythonServiceClient.get()
                .uri("/config")
                .retrieve()
                .bodyToMono(MAP_TYPE_REF);
    }

    /**
     * Send prediction request to Python service
     */
    public Mono<Map<String, Object>> predict(Map<String, Object> request) {
        return pythonServiceClient.post()
                .uri("/predict")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(MAP_TYPE_REF);
    }

    /**
     * Get available models from Python service
     */
    public Mono<Map<String, Object>> getAvailableModels() {
        return pythonServiceClient.get()
                .uri("/models")
                .retrieve()
                .bodyToMono(MAP_TYPE_REF);
    }

    /**
     * Activate a different model
     */
    public Mono<Map<String, Object>> activateModel(Map<String, Object> request) {
        return pythonServiceClient.post()
                .uri("/models/activate")
                .bodyValue(request)
                .retrieve()
                .bodyToMono(MAP_TYPE_REF);
    }
}
