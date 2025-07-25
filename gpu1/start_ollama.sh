#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Set the GPU to use (0 in this case, change if you want to use a different GPU)
GPU_TO_USE=5

# Function to check if a port is in use
port_in_use() {
    lsof -i:$1 > /dev/null 2>&1
}

# Function to check if model exists locally
model_exists() {
    local port=$1
    local model=$2
    OLLAMA_HOST=127.0.0.1:$port ollama list | grep -q "$model"
}

# Function to start Ollama and load model
start_ollama() {
    port=$1
    model="llama3.1:8b"
    log_file="logs/ollama_$port.log"
    
    echo "Starting Ollama on port $port using AMD GPU $GPU_TO_USE..."
    
    # Check if the port is already in use
    if port_in_use $port; then
        echo "Error: Port $port is already in use."
        return 1
    fi
    
    # Start Ollama server using environment variables
    HIP_VISIBLE_DEVICES=$GPU_TO_USE OLLAMA_HOST=127.0.0.1:$port ollama serve > "$log_file" 2>&1 &
    pid=$!
    
    # Wait for the server to start (with timeout)
    timeout=30
    while ! curl -s "http://localhost:$port/api/tags" > /dev/null 2>&1; do
        sleep 1
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            echo "Error: Timed out waiting for Ollama to start on port $port"
            echo "Last few lines of log:"
            tail -n 5 "$log_file"
            kill $pid
            return 1
        fi
        # Check if process is still running
        if ! kill -0 $pid 2> /dev/null; then
            echo "Error: Ollama process on port $port died unexpectedly"
            echo "Last few lines of log:"
            tail -n 5 "$log_file"
            return 1
        fi
    done
    
    echo "Ollama started on port $port. Checking for model $model..."
    
    # Check if the model already exists
    if model_exists $port $model; then
        echo "Model $model already exists locally. Skipping pull."
    else
        echo "Model $model not found locally. Pulling model..."
        # Load the model
        if HIP_VISIBLE_DEVICES=$GPU_TO_USE OLLAMA_HOST=127.0.0.1:$port ollama pull $model >> "$log_file" 2>&1; then
            echo "Successfully pulled $model on port $port"
        else
            echo "Error: Failed to pull $model on port $port"
            echo "Last few lines of log:"
            tail -n 5 "$log_file"
            kill $pid
            return 1
        fi
    fi
}

# Array of ports to use (avoiding conflicts with existing demos)
ports=(11438 11439 11440 11441)

# Start instances sequentially
for port in "${ports[@]}"
do
    if ! start_ollama $port; then
        echo "Failed to start Ollama instance on port $port. Stopping script."
        exit 1
    fi
    sleep 5  # Wait a bit before starting the next instance
done

echo "All Ollama instances have been started successfully on AMD GPU $GPU_TO_USE."
echo "Log files can be found in the 'logs' directory."