CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --served-model-name Qwen3-VL-235B-A22B-Instruct \
  --max_model_len 16384 \
  --tensor-parallel-size 8 \
  --port 23333 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image": 4}'
