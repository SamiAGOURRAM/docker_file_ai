package com.earthquake.controllers;

import com.earthquake.services.PredictionService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.Map;

@Controller
@RequiredArgsConstructor
public class EarthquakeController {

    private final PredictionService predictionService;

    /**
     * Home page
     */
    @GetMapping("/")
    public String homePage(Model model) {
        return "index";
    }

    /**
     * Configuration route - shows model configuration
     */
    @GetMapping("/configuration")
    public String configPage(Model model) {
        Mono<Map<String, Object>> configMono = predictionService.getModelConfig();
        Map<String, Object> config = configMono.block();
        model.addAttribute("config", config);
        return "configuration";
    }

    /**
     * Classification page
     */
    @GetMapping("/classification")
    public String classificationPage(Model model) {
        // Get available models for the dropdown
        Mono<Map<String, Object>> modelsMono = predictionService.getAvailableModels();
        Map<String, Object> modelsInfo = modelsMono.block();
        model.addAttribute("modelsInfo", modelsInfo);
        return "classification";
    }

    /**
     * REST API endpoint for predictions
     */
    @PostMapping(value = "/api/predict", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public Mono<ResponseEntity<Map<String, Object>>> predict(@RequestBody Map<String, Object> request) {
        return predictionService.predict(request)
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(
                        ResponseEntity.badRequest().body(Map.of(
                                "error", e.getMessage(),
                                "status", "error"
                        ))
                ));
    }

    /**
     * REST API endpoint for model configuration
     */
    @GetMapping(value = "/api/config", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public Mono<ResponseEntity<Map<String, Object>>> getConfig() {
        return predictionService.getModelConfig()
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(
                        ResponseEntity.badRequest().body(Map.of(
                                "error", e.getMessage(),
                                "status", "error"
                        ))
                ));
    }
    
    /**
     * REST API endpoint to get available models
     */
    @GetMapping(value = "/api/models", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public Mono<ResponseEntity<Map<String, Object>>> getModels() {
        return predictionService.getAvailableModels()
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(
                        ResponseEntity.badRequest().body(Map.of(
                                "error", e.getMessage(),
                                "status", "error"
                        ))
                ));
    }
    
    /**
     * REST API endpoint to activate a different model
     */
    @PostMapping(value = "/api/models/activate", produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public Mono<ResponseEntity<Map<String, Object>>> activateModel(@RequestBody Map<String, Object> request) {
        return predictionService.activateModel(request)
                .map(ResponseEntity::ok)
                .onErrorResume(e -> Mono.just(
                        ResponseEntity.badRequest().body(Map.of(
                                "error", e.getMessage(),
                                "status", "error"
                        ))
                ));
    }
}