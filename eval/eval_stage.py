import json
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config_schema import EvalConfig
from .eval_client import AestheticEvalClient, EvalClient


class EvalPrompts:
    def __init__(self):
        self.visual_qa_system_prompt = "You are a helpful assistant."
        self.visual_qa_user_prompt = (
            "Question: {}\nOptions:\n{}\n\nBased on the given image only, answer with the option's letter from the given choices directly."
        )


class EvalStage:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.prompts = EvalPrompts()
        self.visual_qa_cli = EvalClient(cfg, self.prompts.visual_qa_system_prompt)
        self.aesthetic_cli = None
        if "aesthetic" in cfg.tasks:
            self.aesthetic_cli = AestheticEvalClient(cfg)

    def run_visual_qa(self, image: Union[Image.Image, str, bytes], qa_list: List[dict]) -> dict:
        correct_count = 0
        total_count = len(qa_list)
        results = []

        for item in tqdm(qa_list, desc="Processing QA"):
            question = item["question"]
            gt_char = str(item["correct_option"]).strip().upper()
            options_dict = item["options"]
            sorted_keys = sorted(options_dict.keys())

            options_text_list = [f"{key}. {options_dict[key]}" for key in sorted_keys]
            options_text = "\n".join(options_text_list)
            user_prompt = self.prompts.visual_qa_user_prompt.format(question, options_text)
            response = self.visual_qa_cli.eval(image, user_prompt)

            pred_char = ""
            if response and "answer" in response:
                pred_char = str(response["answer"]).strip().upper()

            is_correct = pred_char == gt_char
            if is_correct:
                correct_count += 1

            results.append(
                {
                    "qid": item.get("id", "unknown"),
                    "level": item.get("level", "unknown"),
                    "question": question,
                    "options": options_dict,
                    "gt": gt_char,
                    "pred": pred_char,
                    "is_correct": is_correct,
                }
            )

        score = correct_count / total_count if total_count > 0 else 0
        return {"score": score, "details": results}

    def run_aesthetic_quality(self, image_path: str) -> dict:
        if self.aesthetic_cli is None:
            raise RuntimeError("AestheticEvalClient is not initialized because 'aesthetic' is not in eval.tasks.")
        score = self.aesthetic_cli.eval(image_path)
        return {"score": score}

    def run(self, task: str, image: Union[Image.Image, str], prompt: dict) -> dict:
        if task == "visual_qa":
            return self.run_visual_qa(image, prompt["checklist"])
        if task == "aesthetic":
            return self.run_aesthetic_quality(image)
        raise NotImplementedError(f"Task {task} not implemented in EvalStage")

    def load_prompt_id_2_steps(self, records_file: str) -> dict:
        p = Path(records_file)
        if not p.exists():
            return {}
        prompt_id_2_steps = {}
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                prompt_id = obj.get("prompt_id")
                latest = obj.get("steps", [])
                prompt_id_2_steps[prompt_id] = latest
        return prompt_id_2_steps

    def report_scores(self, prompts: List[dict], records_file: str, metric_file: str):
        prompt_id_2_steps = self.load_prompt_id_2_steps(records_file)
        aesthetic_scores = []
        level_stats = {}
        infer_sample_count = 0  # infer 的样本数：每个 prompt_id 一次 visual_qa 评测计为 1

        for prompt in tqdm(prompts, desc="Summarizing scores"):
            prompt_id = prompt["id"]
            eval_steps = prompt_id_2_steps.get(str(prompt_id), [])
            for eval_step in eval_steps:
                task = eval_step.get("task")
                score = eval_step.get("score")
                if task == "aesthetic" and score is not None:
                    if isinstance(score, dict) and "score" in score:
                        aesthetic_scores.append(float(score["score"]))
                    else:
                        try:
                            aesthetic_scores.append(float(score))
                        except (ValueError, TypeError):
                            pass
                elif task == "visual_qa" and score is not None:
                    infer_sample_count += 1
                    details = eval_step.get("details", [])
                    for detail in details:
                        level = detail.get("level", "unknown")
                        is_correct = detail.get("is_correct", False)
                        if level not in level_stats:
                            level_stats[level] = {"correct": 0, "total": 0}
                        level_stats[level]["total"] += 1
                        if is_correct:
                            level_stats[level]["correct"] += 1

        report_dict = {"level_breakdown": {}}
        level_micro_accs = []
        for level, stats in sorted(level_stats.items()):
            total = stats["total"]
            correct = stats["correct"]
            acc = 100 * correct / total if total > 0 else 0.0
            level_micro_accs.append(acc)
            report_dict["level_breakdown"][level] = {"micro_accuracy": acc, "correct": correct, "total": total}

        report_dict["sample_count"] = int(infer_sample_count if infer_sample_count > 0 else 0)

        aesthetic_mean = float(np.mean(aesthetic_scores)) if aesthetic_scores else 0.0
        report_dict["aesthetic"] = {"score": aesthetic_mean, "count": len(aesthetic_scores)}

        final_components = level_micro_accs.copy()
        if aesthetic_scores:
            final_components.append(aesthetic_mean)
        report_dict["final_overall_score"] = float(np.mean(final_components)) if final_components else 0.0

        print(json.dumps(report_dict, indent=4))
        with open(metric_file, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4)
