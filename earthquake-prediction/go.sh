#!/bin/bash

# Set script to exit on error
set -e

echo "=== Earthquake Prediction Application ==="
echo "Starting containers..."

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed or not in PATH"
    exit 1
fi

# Create directories for model architectures
mkdir -p python-service/models/MLP
mkdir -p python-service/models/CNN
mkdir -p python-service/models/RNN
mkdir -p python-service/models/GRU
mkdir -p python-service/models/RandomForest

# Check if model files exist
if [ ! -f "python-service/config.json" ] || [ ! -f "python-service/scaler.joblib" ]; then
    echo "Warning: Model config files not found."
    echo "Please make sure to run your notebook in Colab and export models first."
    
    # Create default config with sampling techniques
    cat > python-service/config.json << EOF
{
  "active_model": {
    "type": "GRU",
    "sampling": "smote"
  },
  "available_models": [
    {
      "type": "MLP",
      "sampling": "original",
      "accuracy": 0.5755,
      "precision": 0.6489,
      "recall": 0.5755,
      "f1_score": 0.6007,
      "roc_auc": 0.5580
    },
    {
      "type": "MLP",
      "sampling": "smote",
      "accuracy": 0.7410,
      "precision": 0.6747,
      "recall": 0.7410,
      "f1_score": 0.6700,
      "roc_auc": 0.5940
    },
    {
      "type": "CNN",
      "sampling": "original",
      "accuracy": 0.7482,
      "precision": 0.5598,
      "recall": 0.7482,
      "f1_score": 0.6404,
      "roc_auc": 0.6824
    },
    {
      "type": "CNN",
      "sampling": "smote",
      "accuracy": 0.7482,
      "precision": 0.7099,
      "recall": 0.7482,
      "f1_score": 0.7118,
      "roc_auc": 0.6313
    },
    {
      "type": "RNN",
      "sampling": "original",
      "accuracy": 0.7410,
      "precision": 0.6747,
      "recall": 0.7410,
      "f1_score": 0.6700,
      "roc_auc": 0.6670
    },
    {
      "type": "RNN",
      "sampling": "combined",
      "accuracy": 0.7626,
      "precision": 0.7417,
      "recall": 0.7626,
      "f1_score": 0.6932,
      "roc_auc": 0.6527
    },
    {
      "type": "GRU",
      "sampling": "original",
      "accuracy": 0.7770,
      "precision": 0.7547,
      "recall": 0.7770,
      "f1_score": 0.7541,
      "roc_auc": 0.7516
    },
    {
      "type": "GRU",
      "sampling": "smote",
      "accuracy": 0.7842,
      "precision": 0.7681,
      "recall": 0.7842,
      "f1_score": 0.7450,
      "roc_auc": 0.7530
    }
  ]
}
EOF
    
    echo "Created default config.json file with sampling variants"
fi

# Check if any model exists
MODEL_COUNT=$(find python-service/models -type f | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "No model files found in python-service/models directory."
    echo "Creating placeholder model files for demonstration."
    
    # Create placeholder model files for each sampling variant in the config
    for MODEL_TYPE in MLP CNN RNN GRU RandomForest; do
        for SAMPLING in original smote under combined; do
            PLACEHOLDER_FILE="python-service/models/$MODEL_TYPE/$SAMPLING.h5"
            if [ "$MODEL_TYPE" == "RandomForest" ]; then
                PLACEHOLDER_FILE="python-service/models/$MODEL_TYPE/$SAMPLING.joblib"
            fi
            
            # Create empty placeholder file
            touch "$PLACEHOLDER_FILE"
            echo "Created placeholder: $PLACEHOLDER_FILE"
        done
    done
    
    # Create placeholder scaler
    if [ ! -f "python-service/scaler.joblib" ]; then
        touch "python-service/scaler.joblib"
        echo "Created placeholder scaler.joblib"
    fi
fi

# Start the application using docker-compose
docker-compose up -d

# Wait for services to be fully up
echo "Waiting for services to start..."
sleep 10

# Check if services are running
if [ "$(docker ps -q -f name=earthquake-spring-boot-service)" ] && [ "$(docker ps -q -f name=earthquake-python-service)" ]; then
    echo "=== Services started successfully! ==="
    echo "Web interface available at: http://localhost:8080"
    echo ""
    echo "Test the application with:"
    echo "  - Positive test: curl -X POST -H \"Content-Type: application/json\" -d @test/positive_test.json http://localhost:8080/api/predict"
    echo "  - Negative test: curl -X POST -H \"Content-Type: application/json\" -d @test/negative_test.json http://localhost:8080/api/predict"
    echo ""
    echo "You can select different models with different sampling techniques in the web interface."
    echo ""
    echo "To stop the application, run: docker-compose down"
else
    echo "Error: One or more services failed to start. Check logs with: docker-compose logs"
    exit 1
fi