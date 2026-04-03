import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from .config_schema import EvalAppConfig
from .eval_stage import EvalStage
from utils.io_utils import ensure_dir, read_prompts
from utils.jsonl_utils import append_snapshot, load_latest_steps
from utils.log_utils import setup_logging

logger = setup_logging(__name__, "app.log")


def _build_prompt_id_to_image(records_file: Optional[str]) -> Dict[str, str]:
    if not records_file:
        return {}
    path = Path(records_file)
    if not path.exists():
        logger.warning("infer_records_file does not exist: %s", records_file)
        return {}

    prompt_id_to_image: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                import json

                item = json.loads(line)
            except Exception:
                continue

            prompt_id = item.get("prompt_id")
            image_path = item.get("image_path")
            status = item.get("status")
            if prompt_id and isinstance(image_path, str) and image_path and status == "success":
                prompt_id_to_image[str(prompt_id)] = image_path
    return prompt_id_to_image


def _resolve_image(cfg: EvalAppConfig, prompt: dict, prompt_id_to_image: Dict[str, str]):
    role = cfg.eval.role
    if role == "blank":
        return Image.new("RGB", (50, 50), color="white")

    if role == "gt":
        image_url = prompt.get("image_url")
        if isinstance(image_url, str) and image_url.strip():
            return image_url.strip()

        image_name = prompt.get("image")
        if cfg.data.gt_image_root and isinstance(image_name, str) and image_name.strip():
            return str(Path(cfg.data.gt_image_root) / image_name.strip())
        return None

    if role == "image_gen":
        return prompt_id_to_image.get(str(prompt.get("id")))

    raise NotImplementedError(f"Unknown eval role: {role}")


def _single_eval_task(cfg: EvalAppConfig, ev: EvalStage, prompt: dict, lock, prompt_id_to_image: Dict[str, str]) -> None:
    prompt_id = prompt.get("id")
    if prompt_id is None:
        return

    eval_steps = load_latest_steps(cfg.data.records_file, str(prompt_id))
    done_tasks = [s.get("task") for s in eval_steps]
    miss_tasks = [t for t in cfg.eval.tasks if t not in done_tasks]
    if not miss_tasks:
        return

    image_input = _resolve_image(cfg, prompt, prompt_id_to_image)
    if image_input is None:
        logger.warning("prompt_id=%s image input not found for role=%s", prompt_id, cfg.eval.role)
        return
    if isinstance(image_input, str) and not image_input.startswith("http") and not Path(image_input).exists():
        logger.warning("prompt_id=%s image path does not exist: %s", prompt_id, image_input)
        return

    t0 = time.monotonic()
    for task in miss_tasks:
        ev_result = ev.run(task, image_input, prompt)
        if ev_result is None:
            continue
        eval_steps.append({"task": task, **ev_result})
        append_snapshot(
            lock,
            cfg.data.records_file,
            str(prompt_id),
            eval_steps,
            extra={"stage": "success", "latency_sec": round(time.monotonic() - t0, 4)},
        )


def run_eval_pipeline(cfg: EvalAppConfig) -> None:
    ensure_dir(cfg.data.out_dir)
    prompts: List[dict] = read_prompts(cfg.data.prompts_file, cfg.data.inline_prompts)
    if not prompts:
        logger.warning("No prompts found. Nothing to evaluate.")
        return

    prompt_id_to_image = _build_prompt_id_to_image(cfg.data.infer_records_file)
    ev = EvalStage(cfg.eval)
    logger.info("Loaded %s prompts for standalone eval pipeline.", len(prompts))

    if cfg.run.parallel_mode == "off":
        lock = nullcontext()
        for prompt in prompts:
            _single_eval_task(cfg, ev, prompt, lock, prompt_id_to_image)
    elif cfg.run.parallel_mode == "thread":
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as ex:
            futures = [ex.submit(_single_eval_task, cfg, ev, p, lock, prompt_id_to_image) for p in prompts]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                fut.result()
    else:
        raise NotImplementedError(f"parallel_mode={cfg.run.parallel_mode} is not supported in eval pipeline.")

    ev.report_scores(prompts, cfg.data.records_file, cfg.data.metric_file)
    logger.info("Eval report saved to %s", cfg.data.metric_file)
