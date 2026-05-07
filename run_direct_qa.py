import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoProcessor
import os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# ================================
# 配置
# ================================
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Demo script")

    # 输入参数
    parser.add_argument("--input", type=str, required=True, help="input file")
    parser.add_argument("--output", type=str, required=True, help="output file")
    parser.add_argument("--model", type=str, required=True, help="output file")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_kn", action="store_true")
    args = parser.parse_args()
    return args


def prepare_inputs_for_vllm(messages, processor):

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}

    if image_inputs is not None:
        mm_data["image"] = image_inputs

    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs
    }

# ================================
# 加载知识
# ================================

def load_knowledge():

    kn_file = "query_rag_kn_part_1.jsonl"

    query2kn = {}
    with open(kn_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            query2kn[data["query"]] = data["content"]

    query2rag = {}
    rag_json = "qwen_rag_kn_part_2.jsonl"
    with open(rag_json, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            query2rag[data["query"]] = data["content"]

    return query2kn, query2rag


def build_prompt(question, options, knowledge=None):

    option_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])

    prompt = f"""
你是一个多模态游戏问答助手。

根据图片和参考知识（如有）回答问题，并从给定选项中选择最正确的一项。

问题：
{question}

选项：
{option_text}
"""

    if knowledge:
        prompt += f"\n参考知识：\n{knowledge}\n"

    prompt += "\n只输出正确选项的完整文本，不要解释。"

    return prompt


# ================================
# benchmark
# ================================
def main(benchmark_file, output_file, MODEL_NAME, USE_KNOWLEDGE, BATCH_SIZE):
    # ================================
    # 加载模型
    # ================================

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=16384,
        tensor_parallel_size=4,
        max_num_seqs=4
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=64,
        top_k=-1,
    )

    if USE_KNOWLEDGE:
        query2kn, query2rag = load_knowledge()

    with open(output_file, "w") as fout:

        with open(benchmark_file) as f:
            lines = f.readlines()

        for i in tqdm(range(0, len(lines), BATCH_SIZE)):

            batch_lines = lines[i:i+BATCH_SIZE]

            batch_imgs = []
            batch_prompts = []
            batch_meta = []

            for line in batch_lines:
                try:
                    data = json.loads(line)

                    query = data["query"]
                    img = data["img"]

                    qa = data["qa"]

                    question = qa["question"]
                    options = qa["options"]
                    answer = qa["answer"]

                    knowledge = None

                    if USE_KNOWLEDGE:
                        kn = query2kn.get(query, "")
                        rag = query2rag.get(query, "")
                        knowledge = kn + "\n" + rag

                    prompt = build_prompt(question, options, knowledge)

                    batch_imgs.append(img)
                    batch_prompts.append(prompt)

                    batch_meta.append(
                        {
                            "query": query,
                            "img": img,
                            "question": question,
                            "options": options,
                            "answer": answer,
                            "prompt": prompt
                        }
                    )

                except Exception as e:
                    print(e)
                    continue

            try:

                inputs = []

                for img, prompt in zip(batch_imgs, batch_prompts):

                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt},
                        ],
                    }]

                    inp = prepare_inputs_for_vllm(messages, processor)

                    inputs.append(inp)

                outputs = llm.generate(inputs, sampling_params=sampling_params)

                preds = [out.outputs[0].text.strip() for out in outputs]

                for meta, pred in zip(batch_meta, preds):

                    pred_correct = meta["answer"] in pred

                    result = {
                        **meta,
                        "model_output": pred,
                        "pred_correct": pred_correct,
                        "model_name": MODEL_NAME,
                        "USE_KNOWLEDGE": f"{USE_KNOWLEDGE}"
                    }

                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                print("batch infer error:", e)
            fout.flush()

if __name__ == "__main__":
    args = parse_args()

    print("input:", args.input)
    print("output:", args.output)
    print("model:", args.model)
    print("use_kn:", args.use_kn)
    
    benchmark_file = args.input
    output_file = args.output
    MODEL_NAME = args.model
    USE_KNOWLEDGE = args.use_kn
    BATCH_SIZE = args.batch_size

    main(benchmark_file, output_file, MODEL_NAME, USE_KNOWLEDGE, BATCH_SIZE)
