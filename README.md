# AIBench

<p align="center">
        💜 <a href="https://deep-kaixun.github.io/aibench-page/"><b>Project Page</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/datasets/lntzm/AIBench">Dataset</a>&nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/ali-vilab/AIBench">Code</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/pdf/2603.28068">Paper</a>&nbsp&nbsp
</p>

AIBench is a benchmark and toolkit for evaluating **academic illustration generation**.  
It focuses on whether modern image generation models can produce paper-ready method/framework figures that are both:

- **logically consistent** with the source paper
- **aesthetically acceptable** as academic illustrations

This repository currently provides two core modules:

- `infer/`: image generation
- `eval/`: standalone evaluation

These modules are designed to work together: first generate images, then evaluate either generated outputs or baseline references.

## Why AIBench

Image generation has improved rapidly, but its ability to produce **ready-to-use academic figures** is still underexplored.
Prior evaluations often rely on direct VLM-to-VLM holistic comparisons, which implicitly assume oracle-level multimodal understanding.
That assumption becomes fragile when method text and diagrams are long, dense, and logically complex.

AIBench addresses this gap with a **VQA-style logic evaluation** plus **model-based aesthetic evaluation**, reducing reliance on a single black-box holistic judgment.

## Features

- **First VQA-based benchmark for this task**: evaluates academic illustration quality through structured question answering
- **Four-level logic assessment**: from low-level components to high-level semantics
- **Aesthetic assessment via VLMs**: complements logic consistency with style/visual quality
- **Decoupled design**: inference and evaluation are separated for easier debugging and reuse
- **JSONL data flow**: structured JSONL input/output for scalable batch processing
- **Multi-role evaluation**: supports generated images (`image_gen`), ground-truth images (`gt`), and blank-image baselines (`blank`)
- **Config-driven workflow**: model backend, parallelism, retries, and paths are all controlled via YAML

## Method Overview

For logic evaluation, AIBench formulates assessment as a sequence of VQA tasks built from a logic graph summarized from paper method text.
Questions are organized into four levels:

1. **Component-Existence Level**
2. **Local-Topology Level**
3. **Phase-Architecture Level**
4. **Global-Semantics Level**

This design provides fine-grained, interpretable signals on where a generated illustration succeeds or fails.
Text rendering quality is also implicitly tested, because many QA items require correctly generated labels/tokens in the figure.

## Dataset Snapshot

- **300** open-access papers from top-tier venues
- **5704** QA pairs for logic and quality checking
- QAs are generated through the pipeline and manually checked by multiple human experts

## Main Findings (from the paper)

- The performance gap between models on academic illustration generation is substantially larger than on common generation tasks.
- Logic correctness and aesthetics are often in tension; improving one may hurt the other.
- Strong reasoning and high-density visual generation are both necessary for this task.
- Test-time scaling strategies on both reasoning and generation sides can significantly improve performance.

## Contributions

1. AIBench: the first VQA-based benchmark for evaluating academic illustration generation with logic and aesthetic dimensions.
2. A scalable QA construction framework: generates high-quality, multi-level questions from summarized logic diagrams.
3. A comprehensive evaluation of SOTA open-/closed-source unified models and T2I models, including test-time scaling analyses.

## Repository Structure

```text
.
├── infer/
│   ├── main.py
│   ├── pipeline.py
│   ├── generate_stage.py
│   ├── image_gen_client.py
│   ├── config_schema.py
│   └── configs/
│       └── example.yaml
├── eval/
│   ├── main.py
│   ├── pipeline.py
│   ├── eval_stage.py
│   ├── eval_client.py
│   ├── config_schema.py
│   └── configs/
│       └── example.yaml
├── utils/
├── scripts/
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies in `requirements.txt` include `pyyaml`, `pydantic`, `Pillow`, `tqdm`, `requests`, `openai`, `dotenv`, and the aesthetic scoring package `unipercept-reward`.


## Workflow

1. Prepare prompt data in JSONL format
2. Run generation with `infer/main.py`
3. Set the generated records path to `data.infer_records_file` in the eval config
4. Run evaluation with `eval/main.py`
5. Check evaluation records and metrics


## Inference: Generate Images

Run:

```bash
python infer/main.py -c infer/configs/example.yaml
```

Key config fields (`infer/configs/example.yaml`):

- `data.prompts_file`: prompt JSONL input (benchmark source data, e.g., `data/AIBench.jsonl`)
- `data.out_dir`: output directory (e.g., `work_dirs/infer/example`)
- `data.records_file`: generated records filename (e.g., `gen_records.jsonl`)
- `image_gen.provider`: generation backend (e.g., `nano_banana`, `wan2.6`)

### Prompt Input (JSONL)

Each sample should include an `id` and the paper method section field `method_text`.

If `data.text_prompt_template` is set, the text will be formatted into that template before being sent to the generation model, for example:

```text
Generate a framework figure based on the given paper method section.
{method_text}
```

Outputs:

- Images: `<out_dir>/gen_images/*.png`
- Records: `<out_dir>/<records_file>`

Record example (one JSON object per line):

```json
{"prompt_id":"123","status":"success","image_path":".../gen_images/123.png","provider":"nano_banana","latency_sec":4.21,"ts":1760000000.0}
```

## Evaluation: Score Results

Run:

```bash
python eval/main.py -c eval/configs/example.yaml
```

Key config fields (`eval/configs/example.yaml`):

- `data.prompts_file`: prompt JSONL (benchmark source data, e.g., `data/AIBench.jsonl`)
- `data.infer_records_file`: infer output records JSONL (with `prompt_id` and `image_path`)
- `data.gt_image_root`: GT image root directory (used when `role=gt`)
- `eval.role`:
  - `image_gen`: evaluate generated images from infer records
  - `gt`: evaluate dataset GT images (`image_url` or `gt_image_root + image`)
  - `blank`: evaluate blank white-image baseline
- `eval.tasks`: evaluation tasks (e.g., `visual_qa`, `aesthetic`)

Outputs:

- Evaluation records: `<out_dir>/<records_file>`
- Aggregated metrics: `<out_dir>/<metric_file>`


## Configuration Notes

- This project reads environment variables via `!env` (for example, `API_KEY_1`)
- Do not commit secrets; inject keys through `.env` or shell environment variables
- Tune `max_workers` and `retry_times` based on your compute resources and rate limits


## Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@article{liao2025aibench,
  title={AIBench: Evaluating Visual-Logical Consistency in Academic Illustration Generation},
  author={Liao, Zhaohe and Jiang, Kaixun and Liu, Zhihang and Wei, Yujie and Yu, Junqiu and Li, Quanhao and Yu, Hongtao and Li, Pandeng and Wang, Yuzheng and Xing, Zhen and Zhang, Shiwei and Xie, Chen-Wei and Zheng, Yun and Liu, Xihui},
  journal={arXiv preprint arXiv:2603.28068},
  year={2026}
}
```
