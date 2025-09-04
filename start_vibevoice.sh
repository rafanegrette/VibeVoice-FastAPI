#!/bin/bash
# Clear CUDA memory cache
#nvidia-smi --gpu-reset
python -c "import torch; torch.cuda.empty_cache()" 

# Start the API server
source ~/anaconda3/bin/activate vibevoice 
python main.py
