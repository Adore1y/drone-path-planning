#!/bin/bash

# Install dependencies for Drone DRL Framework

echo "Installing dependencies for Drone DRL Framework..."

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Install Python dependencies
echo "Installing Python packages..."
pip install torch numpy gym matplotlib pandas

# Check if Webots is installed
if ! command -v webots &> /dev/null; then
    echo "Warning: Webots command not found in PATH."
    echo "Please ensure Webots is installed and available in your PATH for simulation."
    
    # Provide platform-specific installation guidance
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "On macOS, you can install Webots using:"
        echo "  brew install --cask webots"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "On Ubuntu/Debian, you can install Webots using:"
        echo "  sudo apt update"
        echo "  sudo apt install webots"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "On Windows, please download Webots installer from:"
        echo "  https://cyberbotics.com/"
    fi
else
    echo "Webots found: $(which webots)"
    echo "Webots version: $(webots --version)"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p training_results
mkdir -p configs

# Make training script executable
chmod +x run_training.sh

echo "Installation complete!"
echo ""
echo "You can now run the training with:"
echo "  ./run_training.sh"
echo ""
echo "For help and options, run:"
echo "  ./run_training.sh --help" 