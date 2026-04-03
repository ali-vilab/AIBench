from typing import List, Optional, Literal

from pydantic import BaseModel


class RunConfig(BaseModel):
    parallel_mode: Literal["thread", "off"] = "thread"
    max_workers: int = 8


class DataConfig(BaseModel):
    prompts_file: Optional[str] = None
    inline_prompts: List[dict] = []
    infer_records_file: Optional[str] = None
    gt_image_root: Optional[str] = None
    out_dir: str = "work_dirs/eval"
    records_file: str = "eval_records.jsonl"
    metric_file: str = "eval_metric.json"


class EvalServerCfg(BaseModel):
    model: str = "qwen3-vl-235b-a22b-instruct"
    api_base: str
    api_key: Optional[str] = "EMPTY"
    timeout: int = 1800
    temperature: float = 0.0


class EvalConfig(BaseModel):
    role: Literal["gt", "blank", "image_gen"] = "image_gen"
    tasks: List[Literal["visual_qa", "aesthetic"]] = ["visual_qa", "aesthetic"]
    provider: Literal["api"] = "api"
    max_new_tokens: int = 2048
    retry_times: int = 50
    api: List[EvalServerCfg] = [EvalServerCfg(api_base="http://localhost:23333/v1")]


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class EvalAppConfig(BaseModel):
    run: RunConfig
    data: DataConfig
    eval: EvalConfig
    logging: LoggingConfig = LoggingConfig()
