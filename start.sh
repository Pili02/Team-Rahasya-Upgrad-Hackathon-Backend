#!/bin/bash

# AI MindMap Mentor Startup Script
echo "🧠 Starting AI MindMap Mentor..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python version $PYTHON_VERSION is too old. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if Ollama is running
echo "🔍 Checking Ollama connection..."
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollama is not running on localhost:11434"
    echo "💡 Please start Ollama first with: ollama run llama3"
    echo "   Or start it in another terminal and run this script again."
    echo ""
    echo "Starting Ollama now..."
    if command -v ollama &> /dev/null; then
        echo "🚀 Starting Ollama with LLaMA 3..."
        ollama run llama3 &
        OLLAMA_PID=$!
        echo "⏳ Waiting for Ollama to start..."
        sleep 10
        
        # Check if it's running
        if curl -s http://localhost:11434/api/tags &> /dev/null; then
            echo "✅ Ollama is now running"
        else
            echo "❌ Failed to start Ollama. Please start it manually."
            exit 1
        fi
    else
        echo "❌ Ollama is not installed. Please install it first."
        echo "   Visit: https://ollama.ai/"
        exit 1
    fi
else
    echo "✅ Ollama is running"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📚 Upgrading dependencies..."
pip install --upgrade -r requirements.txt

# Create data directory if it doesn't exist
mkdir -p data/chroma_db

# Start the application
echo "🚀 Starting AI MindMap Mentor..."
echo "📖 API documentation will be available at: http://localhost:8000/docs"
echo "🏥 Health check: http://localhost:8000/health"
echo "🧠 Generate mindmap: POST http://localhost:8000/api/v1/generate_mindmap"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Cleanup if Ollama was started by this script
if [ ! -z "$OLLAMA_PID" ]; then
    echo "🛑 Stopping Ollama..."
    kill $OLLAMA_PID
fi
