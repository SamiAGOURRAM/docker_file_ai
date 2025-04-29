package com.earthquake;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.reactive.function.client.WebClient;

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
    
    @Bean
    public WebClient.Builder webClientBuilder() {
        return WebClient.builder();
    }
    
    @Bean
    public WebClient pythonServiceClient(WebClient.Builder webClientBuilder) {
        return webClientBuilder
                .baseUrl("http://python-service:5000")
                .build();
    }
}