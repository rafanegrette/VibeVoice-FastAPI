# 🎙️ VibeVoice FastAPI - Multi-Speaker TTS API

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Original Project](https://img.shields.io/badge/Original-VibeVoice-orange)](https://github.com/microsoft/VibeVoice)

A production-ready FastAPI wrapper for Microsoft's VibeVoice model, enabling high-quality multi-speaker text-to-speech generation through a REST API with status tracking and queue management.

## ✨ Features

- 🎯 **Multi-Speaker TTS**: Generate conversations with up to 4 distinct speakers
- ⚡ **Asynchronous Processing**: Queue-based system handles multiple requests efficiently  
- 📊 **Status Tracking**: Real-time job status and queue position monitoring
- 🔄 **Rate Limiting**: Built-in protection against API abuse (10 requests/minute)
- 🎵 **Voice Presets**: Pre-configured voice samples for immediate use
- 📁 **File Management**: Automatic output file handling and cleanup
- 🐳 **Docker Ready**: Easy deployment with containerization support

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- At least 8GB GPU memory for optimal performance

### Installation

#### Option 1: Standard Installation (Recommended)
```bash
# Create a conda profile
conda create -n vibevoice python=3.11 -y && conda activate vibevoice

# Clone the FastAPI wrapper
git clone https://github.com/rafanegrette/VibeVoice-FastAPI
cd VibeVoice-FastAPI

# Install FastAPI dependencies
pip install -r requirements.txt
pip install -e .
```

### Starting the API Server

```bash
# Basic usage (defaults to port 8500)
python main.py

# Custom host and port
uvicorn main:app --host 0.0.0.0 --port 8500 --reload
```

The API will be available at `http://localhost:8500` with interactive documentation at `http://localhost:8500/docs`.

## 📚 API Documentation

### Available Models

| Model | Context Length | Generation Length | Hugging Face |
|-------|----------------|-------------------|--------------|
| VibeVoice-1.5B | 64K | ~90 min | [microsoft/VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| VibeVoice-7B-Preview | 32K | ~45 min | [WestZhang/VibeVoice-Large-pt](https://huggingface.co/WestZhang/VibeVoice-Large-pt) |

### Endpoints

#### 1. Generate Audio
**POST** `/generate`

Submit a text-to-speech generation request with multiple speakers.

**Request Body:**
```json
{
  "script": "Speaker 1: Hello, how are you today?\nSpeaker 2: I'm doing great, thanks for asking!",
  "speaker_names": ["en-Alice_woman", "en-Carter_man"],
  "cfg_scale": 1.3
}
```

**Response:** Audio file download or error message.

#### 2. Check Status
**GET** `/status/{task_id}`

Monitor the progress of your generation job.

**Response:**
```json
{
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "running",
  "queue_position": 0,
  "generation_time": null
}
```

**Status Values:**
- `queued`: Job is waiting in queue
- `running`: Job is currently being processed
- `completed`: Job finished successfully
- `failed`: Job encountered an error

#### 3. List Available Voices
**GET** `/voices`

Get all available voice presets.

**Response:**
```json
[
  "en-Alice_woman",
  "en-Carter_man", 
  "en-Frank_man",
  "en-Mary_woman_bgm",
  "zh-Bowen_man",
  "zh-Xinran_woman"
]
```

### Script Format

Your script should follow this format:
```
Speaker 1: First person's dialogue here.
Speaker 2: Second person's response.
Speaker 1: More dialogue from first person.
```

**Important Notes:**
- Each speaker line must start with "Speaker" followed by a number
- Speaker numbers should be consistent throughout the script
- Provide voice names in `speaker_names` array matching the speaker order

## 💡 Usage Examples

### Python Client Example
```python
import requests
import time

# Submit generation request
response = requests.post("http://localhost:8000/generate", json={
    "script": "Speaker 1: Welcome to our podcast!\nSpeaker 2: Thanks for having me!",
    "speaker_names": ["en-Alice_woman", "en-Carter_man"], 
    "cfg_scale": 1.3
})

task_id = response.json()["task_id"]
print(f"Task submitted: {task_id}")

# Poll for completion
while True:
    status_response = requests.get(f"http://localhost:8500/status/{task_id}")
    status = status_response.json()["status"]
    
    if status == "completed":
        # Download the result
        audio_response = requests.get(f"http://localhost:8500/result/{task_id}")
        with open("output.wav", "wb") as f:
            f.write(audio_response.content)
        print("Audio saved as output.wav")
        break
    elif status == "failed":
        print("Generation failed")
        break
    else:
        print(f"Status: {status}")
        time.sleep(5)
```

### cURL Examples
```bash
# Submit generation request
curl -X POST "http://localhost:8500/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "script": "Speaker 1: Hello world!\nSpeaker 2: How are you?",
       "speaker_names": ["en-Alice_woman", "en-Carter_man"],
       "cfg_scale": 1.3
     }'



# List voices
curl "http://localhost:8500/voices"
```

## ⚙️ Configuration

### Model Selection
By default, the API uses `microsoft/VibeVoice-1.5B`. To use the 7B model, modify line 165 in `main.py`:
```python
model_path = "WestZhang/VibeVoice-Large-pt"
```

### Voice Directory
Place your custom voice samples in `demo/voices/` directory. Supported format: WAV files.

### Rate Limiting
Current limit: 10 requests per minute per IP. Modify in `main.py`:
```python
@limiter.limit("10/minute")  # Change this value
```

## 🚨 Important Notes

### Chinese Speech Stability
For optimal Chinese speech generation:
- Use English punctuation (commas and periods only)
- Consider using the 7B model for better stability
- Avoid special Chinese quotation marks

### Background Music
The model may spontaneously generate background music:
- Voice samples with BGM increase the likelihood
- Introductory phrases ("Welcome to", "Hello") may trigger BGM
- Using "Alice" voice preset has higher BGM probability
- This is an intentional feature, not a bug

### Memory Requirements
- **1.5B model**: ~8GB GPU memory
- **7B model**: ~16GB GPU memory
- CPU inference is supported but significantly slower

## 📋 System Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   FastAPI       │───▶│  Task Queue  │───▶│ Worker Thread   │
│   Web Server    │    │ (in-memory)  │    │ (GPU Inference) │
└─────────────────┘    └──────────────┘    └─────────────────┘
        │                       │                     │
        ▼                       ▼                     ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Rate Limiter  │    │ Job Tracking │    │ Audio Output    │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

**Key Features:**
- Single-process architecture for simplicity
- In-memory queue ensures FIFO processing
- Background worker prevents blocking the web server
- Status tracking provides real-time updates

## 📄 License & Attribution

This project is a FastAPI wrapper around Microsoft's VibeVoice model. Please refer to the [original VibeVoice repository](https://github.com/microsoft/VibeVoice) for licensing terms and model details.

## ⚠️ Ethical Use & Limitations

**Responsible AI Usage:**
- Disclose AI-generated content when sharing
- Ensure compliance with local laws and regulations  
- Verify content accuracy and avoid misleading applications
- Do not use for deepfakes or disinformation

**Technical Limitations:**
- English and Chinese only
- No overlapping speech generation
- Speech synthesis only (no background noise/music control)
- Not recommended for commercial use without additional testing

**Model is intended for research and development purposes. Use responsibly.**
