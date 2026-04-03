import logging
import random
import requests
import io
import bentoml
from typing import Dict, Optional, List
import time
from PIL import Image
from .config_schema import ImageGenConfig
from utils.img_utils import base64_to_pil, pil_to_base64
from utils.log_utils import setup_logging

logger = setup_logging(__name__, 'app.log')


class ImageGenClient:
    def __init__(self, cfg: ImageGenConfig):
        self.cfg = cfg
        self.server_pools = []
        self.provider = cfg.provider
        if self.provider == 'qwen_image':
            for endpoint in self.cfg.qwen_image:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model
        elif self.provider == 'wan2.5':
            for endpoint in self.cfg.wan_2_5:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model
        elif self.provider == 'wan2.6':
            for endpoint in self.cfg.wan_2_6:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model
        elif self.provider == 'wan2.7':
            logger.warning("⚠️ Wan2.7 API输入token长度大于1k时强制无法开启thinking_mode，需要外接rewrite结果！！！")
            for endpoint in self.cfg.wan_2_7:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model
        elif self.provider in ['nano_banana', 'nano_banana_pro']:
            for endpoint in self.cfg.nano_banana:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model
        elif self.provider == 'gpt-image-1.5':
            for endpoint in self.cfg.gpt_image_1_5:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model
        elif self.provider in ['seedream4.5', 'seedream5.0']:
            for endpoint in self.cfg.seedream:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model
        elif self.provider == 'klingv3':
            for endpoint in self.cfg.kling:
                self.server_pools.append(
                    {
                        "api_url": endpoint.api_base,
                        "model": endpoint.model,
                        "api_key": endpoint.api_key,
                    }
                )
                self.model = endpoint.model

    def wan_post(self, prompt: str) -> Optional[Image.Image | str]:
        task_id, api_key = self.call_wan_submit(prompt)
        if task_id == 'limit':
            raise Exception("超过并发上限，请稍后再试！")
        if task_id == 'inappropriate':
            logger.error("警告：输入数据可能包含不适当内容。请检查输入数据。")
            return 'inappropriate'
        gen_images = self.call_wan_get(task_id, api_key)
        if len(gen_images) == 0:
            return 'inappropriate'
        elif len(gen_images) == 1:
            return gen_images[0]
        return gen_images

    def qwen_bentoml_post(self, prompt: str) -> Optional[Image.Image]:
        server = random.choice(self.server_pools)
        input_args = {
            "prompt": [prompt],
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 8,
            "true_cfg_scale": 1.0,
            "seed": self.cfg.seed,
        }
        try:
            with bentoml.SyncHTTPClient(server['api_base'], timeout=server['timeout']) as client:
                images = client.txt2img(
                    input_args=input_args,
                )
            gen_image = base64_to_pil(images[0])
            return gen_image
        except Exception as e:
            logger.error(f"错误：调用服务失败。URL: {server['api_base']}, 错误信息: {e}")
            return None
    
    def nano_banana_post(self, prompt):
        server = random.choice(self.server_pools)
        payload = {
            "model": server['model'],
            "dashscope_extend_params": {
                "using_native_protocol": True
            },
            "stream": False,
            "contents": [{
                "parts": [{"text": prompt}],
                "role": "user"
            }],
            "generationConfig": {"responseModalities": ["image"]}
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Authorization: {server['api_key']}"
        }
        try:
            response = requests.post(server['api_url'], json=payload, headers=headers, timeout=240)
            response.raise_for_status()
            response_data = response.json()['candidates'][0]['content']['parts']
            image_url = None
            for d in response_data:
                if 'inlineData' in d:
                    image_url = d['inlineData']['data']
                    break
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            # 如果有响应内容，也打印出来，方便排查问题
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            if 'Requests rate limit exceeded' in e.response.text:
                logger.error("超过并发上限，可以重试多次！")
                return None
            elif 'Input data may contain inappropriate content.' in e.response.text:
                logger.error("输入数据可能包含不appropriate内容。请检查输入数据。")
                return None
            elif 'url error' in e.response.text:
                logger.error("url error, retry")
                return None
            else:
                return None
        except Exception as e:
            logger.error(f"错误信息: {e}")
            return None
        
        if image_url is None:
            logger.error("没有找到图片URL。请检查响应数据。")
            return None
        
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            pil_image = Image.open(io.BytesIO(image_response.content))
            return pil_image
        except requests.exceptions.RequestException as e:
            logger.error(f"错误：下载图片失败。URL: {image_url}, 错误信息: {e}")
            return None
        except Exception as e:
            logger.error(f"下载图片发生未知错误: {e}")
            return None

    def gpt_image_post(self, prompt):
        server = random.choice(self.server_pools)
        payload = {
            "model": server['model'],
            "prompt": prompt,
            "background": "opaque",
            "moderation": "auto",
            "n": 1,
            # "output_compression": 90,
            "output_format": "png",
            "quality":"auto",
            # "size":"1024x1024",
            "user":"user123"
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Authorization: {server['api_key']}"
        }
        try:
            response = requests.post(server['api_url'], json=payload, headers=headers, timeout=240)
            response.raise_for_status()
            image_url = response.json()['data'][0]['b64_json']
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            # 如果有响应内容，也打印出来，方便排查问题
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            if 'Requests rate limit exceeded' in e.response.text:
                logger.error("超过并发上限，可以重试多次！")
                return None
            elif 'Input data may contain inappropriate content.' in e.response.text:
                logger.error("输入数据可能包含不appropriate内容。请检查输入数据。")
                return None
            elif 'url error' in e.response.text:
                logger.error("url error, retry")
                return None
            else:
                return None
        except Exception as e:
            logger.error(f"错误信息: {e}")
            return None
        
        if image_url is None:
            logger.error("没有找到图片URL。请检查响应数据。")
            return None
        
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            pil_image = Image.open(io.BytesIO(image_response.content))
            return pil_image
        except requests.exceptions.RequestException as e:
            logger.error(f"错误：下载图片失败。URL: {image_url}, 错误信息: {e}")
            return None
        except Exception as e:
            logger.error(f"下载图片发生未知错误: {e}")
            return None

    def seedream_post(self, prompt):
        server = random.choice(self.server_pools)
        payload = {
            "model": server['model'],
            "prompt": prompt,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {
                "max_images": 1
            },
            "response_format": "url",
            "size": "2K",
            "watermark": False
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Authorization: {server['api_key']}"
        }
        try:
            response = requests.post(server['api_url'], json=payload, headers=headers, timeout=240)
            response.raise_for_status()
            image_url = response.json()['data']['data'][0]['url']
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            # 如果有响应内容，也打印出来，方便排查问题
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            if 'Requests rate limit exceeded' in e.response.text:
                logger.error("超过并发上限，可以重试多次！")
                return None
            elif 'Input data may contain inappropriate content.' in e.response.text:
                logger.error("输入数据可能包含不appropriate内容。请检查输入数据。")
                return None
            elif 'url error' in e.response.text:
                logger.error("url error, retry")
                return None
            else:
                return None
        except Exception as e:
            logger.error(f"错误信息: {e}")
            return None
        
        if image_url is None:
            logger.error("没有找到图片URL。请检查响应数据。")
            return None
        
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            pil_image = Image.open(io.BytesIO(image_response.content))
            return pil_image
        except requests.exceptions.RequestException as e:
            logger.error(f"错误：下载图片失败。URL: {image_url}, 错误信息: {e}")
            return None
        except Exception as e:
            logger.error(f"下载图片发生未知错误: {e}")
            return None

    def qwen_post(self, prompt):
        server = random.choice(self.server_pools)
        payload = {
            "model": server['model'],
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{"text": prompt}]
                }]
            },
            "parameters": {
                # "size": "1024*1024",
                "n": 1,
                "prompt_extend": self.cfg.gen_rewrite,
                "watermark": False,
            }
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Authorization: Bearer {server['api_key']}"
        }
        try:
            response = requests.post(server['api_url'], json=payload, headers=headers, timeout=240)
            response.raise_for_status()
            image_url = response.json()['output']['choices'][0]['message']['content'][0]['image']
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            # 如果有响应内容，也打印出来，方便排查问题
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            if 'Requests rate limit exceeded' in e.response.text:
                logger.error("超过并发上限，可以重试多次！")
                return None
            elif 'Input data may contain inappropriate content.' in e.response.text:
                logger.error("输入数据可能包含不appropriate内容。请检查输入数据。")
                return None
            elif 'url error' in e.response.text:
                logger.error("url error, retry")
                return None
            else:
                return None
        except Exception as e:
            logger.error(f"错误信息: {e}")
            return None
        
        if image_url is None:
            logger.error("没有找到图片URL。请检查响应数据。")
            return None
        
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            pil_image = Image.open(io.BytesIO(image_response.content))
            return pil_image
        except requests.exceptions.RequestException as e:
            logger.error(f"错误：下载图片失败。URL: {image_url}, 错误信息: {e}")
            return None
        except Exception as e:
            logger.error(f"下载图片发生未知错误: {e}")
            return None

    def kling_post(self, prompt):
        task_id, api_key = self.call_kling_submit(prompt)
        if task_id == 'limit':
            raise Exception("超过并发上限，请稍后再试！")
        if task_id == 'inappropriate':
            logger.error("警告：输入数据可能包含不适当内容。请检查输入数据。")
            return 'inappropriate'
        gen_image = self.call_kling_get(task_id, api_key)
        if gen_image is None:
            return 'inappropriate'
        return gen_image

    def generate(self, prompt: str) -> Image.Image | str:
        max_retries = self.cfg.retry_times
        initial_delay = 2
        backoff_factor = 1
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                if self.provider in ['wan2.5', 'wan2.6', 'wan2.7']:
                    gen_image = self.wan_post(prompt)
                elif self.provider in ['qwen_image', 'qwen_image_max', 'qwen_image_plus']:
                    gen_image = self.qwen_post(prompt)
                elif self.provider in ['nano_banana', 'nano_banana_pro']:
                    gen_image = self.nano_banana_post(prompt)
                elif self.provider == 'gpt-image-1.5':
                    gen_image = self.gpt_image_post(prompt)
                elif self.provider in ['seedream4.5', 'seedream5.0']:
                    gen_image = self.seedream_post(prompt)
                elif self.provider == 'klingv3':
                    gen_image = self.kling_post(prompt)
                else:
                    raise ValueError("Invalid provider")

                if isinstance(gen_image, Image.Image):
                    return gen_image
                elif isinstance(gen_image, List) and isinstance(gen_image[0], Image.Image):
                    return gen_image
                else:
                    raise Exception("Failed to get answer")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} error: {e}")
                if attempt == max_retries - 1:
                    return 'inappropriate'
                time.sleep(delay)
                delay *= backoff_factor
                delay = min(delay, 10)
        # raise Exception(f"Failed to get answer after {max_retries} attempts")
        return "inappropriate"


    def call_wan_submit(self, prompt):
        if self.provider == 'wan2.5':
            input_args = {
                "model": self.model,
                "input": {"prompt": prompt},
                "parameters": {
                    # "size": "1024*1024",
                    "n": self.cfg.return_n,
                    "prompt_extend": False,
                    "watermark": False,
                }
            }
        elif self.provider == 'wan2.6':
            input_args = {
                "model": self.model,
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [{"text": prompt}]
                    }]
                },
                "parameters": {
                    # "size": "1024*1024",
                    "n": self.cfg.return_n,
                    "prompt_extend": self.cfg.gen_rewrite,
                    "watermark": False,
                }
            }
        elif self.provider == 'wan2.7':
            # wan2.7 API requires n=1; multi-image in one task is not supported.
            input_args = {
                "model": self.model,
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [{"text": prompt}]
                    }]
                },
                "parameters": {
                    "size": "2K",
                    "n": 1,
                    "watermark": False,
                    "thinking_mode": False, # Wan2.7API输入token长度大于1k时无法开启thinking_mode
                }
            }
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        server = random.choice(self.server_pools)
        headers = {
            'X-DashScope-Async': 'enable',
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {server['api_key']}"
        }
        try:
            response = requests.post(server['api_url'], json=input_args, headers=headers)
            response.raise_for_status()
            task_id = response.json()['output']['task_id']
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            # 如果有响应内容，也打印出来，方便排查问题
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            if 'Requests rate limit exceeded' in e.response.text:
                logger.error("超过并发上限，可以重试多次！")
                return 'limit', None
            elif 'Input data may contain inappropriate content.' in e.response.text:
                logger.error("输入数据可能包含不appropriate内容。请检查输入数据。")
                return 'inappropriate', None
            elif 'url error' in e.response.text:
                logger.error("url error, retry")
                return 'limit', None
            task_id = None
        return task_id, server['api_key']


    def call_wan_get(self, task_id, api_key):
        task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {'Authorization': f"Bearer {api_key}"}
        for _ in range(10000):
            try:
                response = requests.get(task_url, headers=headers, timeout=30)
                response.raise_for_status()
                response_data = response.json()
                if response_data['output']['task_status'] in ['PENDING', 'RUNNING']:
                    logger.info(response_data['output']['task_status'])
                    time.sleep(8)
                    continue
                logger.info(response_data['output']['task_status'])
                if response_data['output']['task_status'] == 'SUCCEEDED':
                    image_urls = []
                    if self.provider == 'wan2.5':
                        for result in response_data['output']['results']:
                            image_urls.append(result['url'])
                    elif self.provider in ['wan2.6', 'wan2.7']:
                        for result in response_data['output']['choices']:
                            image_urls.append(result['message']['content'][0]['image'])
                    else:
                        raise ValueError(f"Unsupported provider: {self.provider}")
                else:
                    image_urls = []
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"\n请求失败: {e}")
                # 如果有响应内容，也打印出来，方便排查问题
                if e.response is not None:
                    logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
                if 'Requests rate limit exceeded' in e.response.text:
                    logger.error("超过并发上限，延时重试！")
                    time.sleep(12)
                    continue
                else:
                    image_url = None
                    break
        
        if len(image_urls) == 0:
            return []
        
        try:
            pil_images = []
            for image_url in image_urls:
                image_response = requests.get(image_url, timeout=30)
                image_response.raise_for_status()
                pil_image = Image.open(io.BytesIO(image_response.content))
                pil_images.append(pil_image)
            return pil_images
        except requests.exceptions.RequestException as e:
            logger.error(f"错误：下载图片失败。URL: {image_url}, 错误信息: {e}")
            return []
        except Exception as e:
            logger.error(f"下载图片发生未知错误: {e}")
            return []


    def call_kling_submit(self, prompt):
        server = random.choice(self.server_pools)
        headers = {
            'X-DashScope-Async': 'enable',
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {server['api_key']}"
        }
        payload = {
            "model": server['model'],
            "input": {
                "prompt": prompt,
                "n": 1
            },
            "parameters": {}
        }
        try:
            response = requests.post(server['api_url'], json=payload, headers=headers)
            response.raise_for_status()
            task_id = response.json()['output']['task_id']
        except requests.exceptions.RequestException as e:
            logger.error(f"\n请求失败: {e}")
            # 如果有响应内容，也打印出来，方便排查问题
            if e.response is not None:
                logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
            if 'Requests rate limit exceeded' in e.response.text:
                logger.error("超过并发上限，可以重试多次！")
                return 'limit', None
            elif 'Input data may contain inappropriate content.' in e.response.text:
                logger.error("输入数据可能包含不appropriate内容。请检查输入数据。")
                return 'inappropriate', None
            elif 'url error' in e.response.text:
                logger.error("url error, retry")
                return 'limit', None
            task_id = None
        return task_id, server['api_key']

    def call_kling_get(self, task_id, api_key):
        task_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {'Authorization': f"Bearer {api_key}"}
        for _ in range(10000):
            try:
                response = requests.get(task_url, headers=headers, timeout=30)
                response.raise_for_status()
                response_data = response.json()
                if response_data['output']['task_status'] in ['PENDING', 'RUNNING']:
                    logger.info(response_data['output']['task_status'])
                    time.sleep(8)
                    continue
                logger.info(response_data['output']['task_status'])
                if response_data['output']['task_status'] == 'SUCCEEDED':
                    image_url = response_data['output']['data']['data']['task_result']['images'][0]['url']
                else:
                    image_url = None
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"\n请求失败: {e}")
                # 如果有响应内容，也打印出来，方便排查问题
                if e.response is not None:
                    logger.error(f"状态码: {e.response.status_code}, 响应内容: {e.response.text}")
                if 'Requests rate limit exceeded' in e.response.text:
                    logger.error("超过并发上限，延时重试！")
                    time.sleep(12)
                    continue
                else:
                    image_url = None
                    break
        
        if image_url is None:
            return None
        
        try:
            image_response = requests.get(image_url, timeout=30)
            image_response.raise_for_status()
            pil_image = Image.open(io.BytesIO(image_response.content))
            return pil_image
        except requests.exceptions.RequestException as e:
            logger.error(f"错误：下载图片失败。URL: {image_url}, 错误信息: {e}")
            return None
        except Exception as e:
            logger.error(f"下载图片发生未知错误: {e}")
            return None