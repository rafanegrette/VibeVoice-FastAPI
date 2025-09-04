import os
import re
import time
import uuid
import queue
import threading
from typing import List, Tuple, Dict, Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# VibeVoice imports
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from transformers.utils import logging

# --- Setup & Configuration ---

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# --- Rate Limiting & Task Queue ---

limiter = Limiter(key_func=get_remote_address)
# An in-memory queue to hold incoming generation requests.
task_queue = queue.Queue()

# --- Global Variables & Model Loading ---

model: Optional[VibeVoiceForConditionalGenerationInference] = None
processor: Optional[VibeVoiceProcessor] = None
voice_mapper: Optional[Any] = None
tasks: Dict[str, Dict[str, Any]] = {}

# --- Helper Classes (Unchanged) ---

class VoiceMapper:
    def __init__(self, base_path: str = "demo/voices"):
        self.voices_dir = base_path
        self.voice_presets = {}
        self.available_voices = {}
        self.setup_voice_presets()

    def setup_voice_presets(self):
        if not os.path.exists(self.voices_dir):
            logger.warning(f"Voices directory not found at {self.voices_dir}")
            return
        wav_files = [f for f in os.listdir(self.voices_dir) if f.lower().endswith('.wav')]
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            self.voice_presets[name] = os.path.join(self.voices_dir, wav_file)
        self.available_voices = {n: p for n, p in self.voice_presets.items() if os.path.exists(p)}
        logger.info(f"Found {len(self.available_voices)} voices in {self.voices_dir}")

    def get_voice_path(self, speaker_name: str) -> str:
        # Exact and partial matching logic
        if speaker_name in self.available_voices: return self.available_voices[speaker_name]
        speaker_lower = speaker_name.lower()
        for name, path in self.available_voices.items():
            if name.lower() in speaker_lower or speaker_lower in name.lower():
                return path
        if self.available_voices: return list(self.available_voices.values())[0]
        raise ValueError("No voice presets available.")

def parse_txt_script(txt_content: str) -> Tuple[List[str], List[str]]:
    lines = txt_content.strip().split('\n')
    scripts, speaker_numbers = [], []
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    current_speaker, current_text = None, ""
    for line in lines:
        line = line.strip()
        if not line: continue
        match = re.match(speaker_pattern, line, re.IGNORECASE)
        if match:
            if current_speaker is not None and current_text:
                scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
                speaker_numbers.append(current_speaker)
            current_speaker, current_text = match.group(1).strip(), match.group(2).strip()
        elif current_text:
            current_text += " " + line
    if current_speaker is not None and current_text:
        scripts.append(f"Speaker {current_speaker}: {current_text.strip()}")
        speaker_numbers.append(current_speaker)
    return scripts, speaker_numbers

# --- Background Worker for Inference ---

def generation_worker():
    """
    A dedicated worker thread that continuously processes tasks from the queue.
    """
    logger.info("Generation worker started.")
    while True:
        task_id, script_content, speaker_names, cfg_scale = task_queue.get()
        if task_id is None: # A way to stop the worker thread if needed.
            break
        
        try:
            logger.info(f"Worker picked up task {task_id}.")
            tasks[task_id]["status"] = "running"
            start_time = time.time()

            # --- Core Inference Logic ---
            scripts, speaker_numbers = parse_txt_script(script_content)
            if not scripts: raise ValueError("No valid scripts found in the provided text.")

            unique_speakers = sorted(list(set(speaker_numbers)), key=int)
            if len(speaker_names) < len(unique_speakers):
                raise ValueError(f"Script has {len(unique_speakers)} speakers, but only {len(speaker_names)} names were given.")

            name_map = {num: name for num, name in zip(unique_speakers, speaker_names)}
            voice_samples = [voice_mapper.get_voice_path(name_map[num]) for num in unique_speakers]
            
            inputs = processor(
                text=['\n'.join(scripts)],
                voice_samples=[voice_samples],
                padding=True, return_tensors="pt", return_attention_mask=True
            )

            outputs = model.generate(
                **inputs, max_new_tokens=None, cfg_scale=cfg_scale,
                tokenizer=processor.tokenizer, generation_config={'do_sample': False},
                verbose=False
            )
            # --- End Core Inference Logic ---
            
            generation_time = time.time() - start_time
            logger.info(f"Task {task_id} finished in {generation_time:.2f}s")
            
            output_dir = "api_outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{task_id}.wav")
            processor.save_audio(outputs.speech_outputs[0], output_path=output_path)
            
            tasks[task_id].update({
                "status": "completed",
                "result_path": output_path,
                "generation_time": generation_time
            })
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            if task_id in tasks:
                tasks[task_id].update({"status": "failed", "error": str(e)})
        finally:
            task_queue.task_done()

# --- FastAPI App Definition ---

app = FastAPI(title="VibeVoice API", description="API for long-form multi-speaker TTS")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.on_event("startup")
async def startup_event():
    global model, processor, voice_mapper
    logger.info("Application startup: loading models...")
    
    model_path = "microsoft/VibeVoice-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        voice_mapper = VoiceMapper(base_path="demo/voices")
        processor = VibeVoiceProcessor.from_pretrained(model_path)
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="sdpa"
        )
        model.eval()
        model.set_ddpm_inference_steps(num_steps=10)
    
    # Start the background worker thread
    worker_thread = threading.Thread(target=generation_worker, daemon=True)
    worker_thread.start()
    
    logger.info("Models loaded and worker thread started.")

# --- Pydantic Models for API I/O ---

class GenerationRequest(BaseModel):
    script: str = Field(..., description="The full script, e.g., 'Speaker 1: ...\nSpeaker 2: ...'")
    speaker_names: List[str] = Field(..., description="List of voice preset names, e.g., ['en-Alice_woman']")
    cfg_scale: float = Field(1.3, ge=1.0, le=2.0)

class GenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str
    queue_position: int

class TaskStatus(BaseModel):
    task_id: str
    status: str
    queue_position: Optional[int] = None
    error: Optional[str] = None
    result_path: Optional[str] = None
    generation_time: Optional[float] = None

# --- API Endpoints ---

@app.post("/generate", response_model=GenerationResponse, status_code=202)
@limiter.limit("10/minute")
async def generate_audio(request: Request, generation_request: GenerationRequest):
    task_id = str(uuid.uuid4())
    queue_pos = task_queue.qsize() + 1
    tasks[task_id] = {"status": "queued", "queue_position": queue_pos}
    
    task_queue.put((
        task_id,
        generation_request.script,
        generation_request.speaker_names,
        generation_request.cfg_scale
    ))
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Job accepted and placed in queue.",
        "queue_position": queue_pos
    }

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Dynamically update queue position if still queued
    if task["status"] == "queued":
        try:
            # Find the position in the current queue
            all_queued_tasks = list(task_queue.queue)
            pos = [item[0] for item in all_queued_tasks].index(task_id) + 1
            task["queue_position"] = pos
        except ValueError:
            # Task might have just been picked up, status will update soon
            task["queue_position"] = 0
            
    return {"task_id": task_id, **task}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task = tasks.get(task_id)
    if not task: raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task not complete. Status: {task['status']}")
    result_path = task.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(result_path, media_type="audio/wav", filename=os.path.basename(result_path))

@app.get("/voices", response_model=List[str])
async def list_voices():
    if not voice_mapper: raise HTTPException(status_code=503, detail="VoiceMapper not initialized.")
    return list(voice_mapper.available_voices.keys())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)