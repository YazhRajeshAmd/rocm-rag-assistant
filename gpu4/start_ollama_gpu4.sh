#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to start Ollama for a specific model
start_ollama() {
    model=$1
    log_file="logs/ollama_${model}.log"
    
    echo "Starting Ollama for model $model..."
    
    ollama serve > "$log_file" 2>&1 &
    pid=$!
    
    # Wait for the server to start (with timeout)
    timeout=30
    while ! pgrep -f "ollama serve" > /dev/null; do
        sleep 1
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            echo "Error: Timed out waiting for Ollama to start for model $model"
            echo "Last few lines of log:"
            tail -n 5 "$log_file"
            return 1
        fi
    done
    
    echo "Ollama started successfully for model $model"
    
    # Run the model
    echo "Running model $model..."
    ollama run $model "Hello" >> "$log_file" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Model $model is running successfully"
    else
        echo "Error: Failed to run model $model"
        echo "Last few lines of log:"
        tail -n 5 "$log_file"
        return 1
    fi
}

# Array of models to use
models=("mixtral:8x7b" "llama3.1:8b" "gemma2:27b" "phi3:14b")

# Start instances sequentially
for model in "${models[@]}"
do
    if ! start_ollama $model; then
        echo "Failed to start Ollama instance for $model. Continuing to next model."
    fi
    sleep 5  # Wait a bit before starting the next instance
done

echo "Attempt to start all Ollama instances completed."
echo "Log files can be found in the 'logs' directory."