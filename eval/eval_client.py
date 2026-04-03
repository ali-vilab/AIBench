import base64
import logging
import os
import random
import re
import tempfile
import time
from typing import List, Union

import openai
import requests
import torch
from openai import OpenAI
from PIL import Image
from unipercept_reward import UniPerceptRewardInferencer

from .config_schema import EvalConfig
from utils.img_utils import pil_to_base64
from utils.log_utils import setup_logging

logger = setup_logging(__name__, "app.log")


class EvalClient:
    def __init__(self, cfg: EvalConfig, system_prompt: str, check_keys: List[str] = [], strict_check: bool = True):
        self.cfg = cfg
        self.system_prompt = system_prompt
        self.max_new_tokens = cfg.max_new_tokens
        self.server_pools = []
        self.check_keys = check_keys
        self.strict_check = strict_check
        for endpoint in self.cfg.api:
            self.server_pools.append(
                {
                    "model": endpoint.model,
                    "api_url": endpoint.api_base,
                    "api_key": endpoint.api_key,
                    "temperature": endpoint.temperature,
                    "timeout": endpoint.timeout,
                }
            )

    def eval(self, image: Union[Image.Image, str, bytes], user_prompt: str) -> dict:
        if isinstance(image, str) and image.startswith("http"):
            base64_image = image
        elif isinstance(image, Image.Image):
            base64_image = pil_to_base64(image)
            base64_image = f"data:image;base64,{base64_image}"
        elif isinstance(image, str) and os.path.isfile(image):
            base64_image = pil_to_base64(Image.open(image).convert("RGB"))
            base64_image = f"data:image;base64,{base64_image}"
        elif isinstance(image, bytes):
            encoded_image = base64.b64encode(image)
            encoded_image_text = encoded_image.decode("utf-8")
            base64_image = f"data:image;base64,{encoded_image_text}"
        else:
            raise ValueError("Invalid image input for EvalClient")

        message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_image}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self.get_answer_with_retry(message, max_retries=self.cfg.retry_times)

    def get_answer_with_retry(self, history, max_retries=5, initial_delay=1, backoff_factor=2):
        delay = initial_delay
        attempt = 0
        while True:
            try:
                response_content, error_code = self.curl_func(history)

                if response_content is None:
                    if attempt >= max_retries - 1:
                        break

                    if error_code == "Unknown":
                        attempt += 1
                    else:
                        attempt += 0.01

                    logger.error("Error code %s, retrying...", error_code)
                    time.sleep(delay)
                    continue

                struct_response_content = self.parse_response(response_content)
                return struct_response_content
            except Exception as e:
                logger.error("Attempt %s error: %s", attempt + 1, e)
                attempt += 1
                if attempt == max_retries - 1:
                    break
                time.sleep(delay)

        logger.error("Failed to get answer after %s attempts", max_retries)
        return {"answer": ""}

    def parse_response(self, response):
        if "<think>" in response:
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response.strip()
        response = response.strip(".").upper()
        if response in ["A", "B", "C", "D"]:
            return {"answer": response}
        if response and response[0] in ["A", "B", "C", "D"]:
            return {"answer": response[0]}
        if response and response[-1] in ["A", "B", "C", "D"]:
            return {"answer": response[-1]}
        raise ValueError(f"Invalid response: {response}")

    def curl_func(self, history):
        try:
            server = random.choice(self.server_pools)
            client = OpenAI(api_key=server["api_key"], base_url=server["api_url"])
            messages = history
        except Exception as e:
            logging.exception(e)
            return None, "Unknown"

        try:
            chat_response = client.chat.completions.create(
                model=server["model"],
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=server["temperature"],
                timeout=server["timeout"],
            )
        except (openai.BadRequestError, openai.RateLimitError, openai.APIStatusError) as e:
            error_json = e.response.json()
            code = error_json["error"].get("code")
            return None, code
        except Exception as e:
            logger.error(e)
            return None, "Unknown"

        try:
            response_content = chat_response.choices[0].message.content
            return response_content, None
        except Exception as e:
            logger.error(e)
            return None, "Unknown"


class AestheticEvalClient:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.model = UniPerceptRewardInferencer(device="cuda")

    def eval(self, image: Union[Image.Image, str, bytes]) -> dict:
        temp_path = None

        if isinstance(image, str) and image.startswith("http"):
            try:
                response = requests.get(image, timeout=30)
                response.raise_for_status()
            except Exception as e:
                logger.error("Failed to process image url: %s", e)
                return {"score": 0.0}
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(response.content)
                temp_path = tmp.name
                image_path = temp_path
        elif isinstance(image, Image.Image):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                temp_path = tmp.name
                image_path = temp_path
        elif isinstance(image, str) and os.path.isfile(image):
            image_path = image
        elif isinstance(image, bytes):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(image)
                temp_path = tmp.name
                image_path = temp_path
        else:
            raise ValueError("Invalid image input")

        try:
            score = 0.0
            with torch.inference_mode():
                rewards = self.model.reward(image_paths=[image_path])
                if rewards and rewards[0]:
                    score_dict = rewards[0]
                    score = score_dict.get("iaa", 0.0)
            return score
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
