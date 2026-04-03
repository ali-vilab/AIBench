from typing import List, Optional, Literal

from pydantic import BaseModel



class RunConfig(BaseModel):
    parallel_mode: Literal["thread", "off"] = "thread"
    max_workers: int = 8


class DataConfig(BaseModel):
    prompts_file: Optional[str] = None
    inline_prompts: List[dict] = []
    text_prompt_template: str = ""
    out_dir: str = "outputs/infer"
    exist_strategy: Literal["skip", "overwrite"] = "skip"
    records_file: str = "gen_records.jsonl"


class GenProvider(BaseModel):
    model: str = ""
    api_base: str
    api_key: Optional[str] = None
    timeout: int = 60

class ImageGenConfig(BaseModel):
    provider: str = "wan2.6"
    wan_2_5: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"),
    ]
    wan_2_6: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"),
    ]
    wan_2_7: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"),
    ]
    qwen_image: List[GenProvider] = [
        GenProvider(api_base=" https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"),
    ]
    nano_banana: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ]
    gemini_flash25_image: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ]
    gpt_image_1_5: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ]
    seedream: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
    ]
    kling: List[GenProvider] = [
        GenProvider(api_base="https://dashscope.aliyuncs.com/api/v1/services/aigc/model-evaluation/async-inference/"),
    ]
    gen_rewrite: bool = False
    seed: int = 42
    retry_times: int = 30
    return_n : int = 1


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class InferConfig(BaseModel):
    run: RunConfig
    data: DataConfig
    image_gen: ImageGenConfig
    logging: LoggingConfig = LoggingConfig()