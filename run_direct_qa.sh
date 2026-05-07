benchmark_file=query2QA.jsonl
output_file=query2QA_result.jsonl
MODEL_NAME=Qwen2.5-VL-7B-Instruct

python run_direct_qa.py \
    --input $benchmark_file \
    --output $output_file \
    --model $MODEL_NAME \
    --batch_size 1 \
    # --use_kn
