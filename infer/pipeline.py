import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import List

from tqdm import tqdm

from infer.config_schema import InferConfig
from .generate_stage import GenerateStage
from utils.io_utils import ensure_dir, read_prompts
from utils.log_utils import setup_logging

logger = setup_logging(__name__, "app.log")


def _resolve_prompt_text(prompt: dict, text_prompt_template: str = "") -> str:
    method_text = prompt.get("method_text")
    if not isinstance(method_text, str) or not method_text.strip():
        return ""

    method_text = method_text.strip()
    if text_prompt_template:
        try:
            return text_prompt_template.format(method_text=method_text)
        except Exception as e:
            logger.warning("Failed to format text_prompt_template, fallback to raw method_text. error=%s", e)
    return method_text


def _append_record(lock, record_path: Path, record: dict) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with record_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_existing_success_prompt_ids(record_path: Path) -> set[str]:
    if not record_path.exists():
        return set()

    success_ids: set[str] = set()
    with record_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("status") == "success":
                prompt_id = obj.get("prompt_id")
                if prompt_id is not None:
                    success_ids.add(str(prompt_id))
    return success_ids


def _single_generate_task(cfg: InferConfig, generate_stage: GenerateStage, prompt: dict, lock) -> None:
    prompt_id = prompt.get("id")
    if prompt_id is None:
        raise ValueError("Each prompt must contain an 'id' field.")

    prompt_text = _resolve_prompt_text(prompt, cfg.data.text_prompt_template)
    if not prompt_text:
        logger.warning("prompt_id=%s has empty prompt text, skip.", prompt_id)
        return

    start_ts = time.monotonic()
    gen_result = generate_stage.run(str(prompt_id), prompt_text)
    latency_sec = round(time.monotonic() - start_ts, 4)

    record = {
        "prompt_id": str(prompt_id),
        "provider": cfg.image_gen.provider,
        "prompt": prompt_text,
        "latency_sec": latency_sec,
        "ts": time.time(),
    }

    if isinstance(gen_result, Path):
        record["status"] = "success"
        record["image_path"] = str(gen_result)
        logger.info("Generated prompt_id=%s -> %s", prompt_id, gen_result)
    elif isinstance(gen_result, str) and gen_result not in {"error", "inappropriate"}:
        # BoN path list: "a.png;b.png;..."
        record["status"] = "success"
        record["image_path"] = gen_result
        logger.info("Generated prompt_id=%s -> %s", prompt_id, gen_result)
    else:
        record["status"] = "failed"
        record["error"] = gen_result
        logger.error("Generate failed for prompt_id=%s, error=%s", prompt_id, gen_result)

    _append_record(lock, Path(cfg.data.records_file), record)


def run_generate_pipeline(cfg: InferConfig) -> None:
    ensure_dir(cfg.data.out_dir)
    prompts: List[dict] = read_prompts(cfg.data.prompts_file, cfg.data.inline_prompts)
    if not prompts:
        logger.warning("No prompts found. Nothing to generate.")
        return

    logger.info("Loaded %s prompts for pure-generate infer pipeline.", len(prompts))
    existing_success_ids = _load_existing_success_prompt_ids(Path(cfg.data.records_file))
    if existing_success_ids:
        prompts = [p for p in prompts if str(p.get("id")) not in existing_success_ids]
        logger.info(
            "Skip %s prompts already recorded as success; remaining=%s.",
            len(existing_success_ids),
            len(prompts),
        )
    if not prompts:
        logger.info("All prompts already exist in records_file with success status. Nothing to generate.")
        return

    generate_stage = GenerateStage(cfg.image_gen, cfg.data.out_dir, cfg.data.exist_strategy)

    if cfg.run.parallel_mode == "off":
        lock = nullcontext()
        for prompt in prompts:
            _single_generate_task(cfg, generate_stage, prompt, lock)
        return

    if cfg.run.parallel_mode != "thread":
        raise NotImplementedError(f"parallel_mode={cfg.run.parallel_mode} is not supported in infer pipeline.")

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as ex:
        futures = [ex.submit(_single_generate_task, cfg, generate_stage, prompt, lock) for prompt in prompts]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            fut.result()
