from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenLogitFilter:
    """
    Qwen2.5-1.5B-Instructを用いた選択肢Logitルーター。
    自己回帰デコードループをバイパスし、単一フォワードパスで充足確率を算定する。
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(device).eval()

        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("No")

        assert self.yes_token_id != self.tokenizer.unk_token_id, "Token 'Yes' not found."
        assert self.no_token_id != self.tokenizer.unk_token_id, "Token 'No' not found."

        self.baseline_margin = 0.0

    @torch.inference_mode()
    def calibrate(self, empty_query: str = "", empty_document: str = "") -> float:
        """Measure baseline positivity bias for calibration"""
        prompt = self._build_prompt(empty_query, empty_document)
        encoded = self.tokenizer(
            [prompt],
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        next_token_logits = outputs.logits[:, -1, :]

        yes_logit = next_token_logits[0, self.yes_token_id].item()
        no_logit = next_token_logits[0, self.no_token_id].item()
        
        self.baseline_margin = float(yes_logit - no_logit)
        return self.baseline_margin

    def _build_prompt(self, query: str, document: str) -> str:
        """ChatML仕様に厳格に準拠した情報充足性判定プロンプトを構築"""
        return (
            "<|im_start|>system\n"
            "You are an information sufficiency validator. Determine whether the provided "
            "document contains explicit information to directly answer the query. "
            "Answer only with 'Yes' or 'No'.<|im_end|>\n"
            "<|im_start|>user\n"
            f"[Query]\n{query}\n\n"
            f"[Document]\n{document}\n\n"
            "Does the document contain sufficient information to answer the query? "
            "Answer 'Yes' or 'No':<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @torch.inference_mode()
    def filter_documents(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        全候補パッセージを一括バッチ処理し、充足確率が閾値以上の文書のみを抽出・再ソートする
        """
        if not candidates:
            return []

        prompts = [self._build_prompt(query, c["text"]) for c in candidates]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        next_token_logits = outputs.logits[:, -1, :]

        yes_logits = next_token_logits[:, self.yes_token_id]
        no_logits = next_token_logits[:, self.no_token_id]

        # Apply baseline calibration to yes_logits
        calibrated_yes_logits = yes_logits - self.baseline_margin

        stacked_logits = torch.stack([no_logits, calibrated_yes_logits], dim=1) / temperature
        probs = F.softmax(stacked_logits, dim=-1)
        yes_probs = probs[:, 1].tolist()

        filtered_results = []
        for candidate, p_yes, z_yes, z_no in zip(
            candidates, yes_probs, yes_logits.tolist(), no_logits.tolist()
        ):
            if p_yes >= threshold:
                item = dict(candidate)
                item["sufficiency_prob"] = float(p_yes)
                item["logit_margin"] = float((z_yes - self.baseline_margin) - z_no)
                filtered_results.append(item)

        filtered_results.sort(key=lambda x: x["sufficiency_prob"], reverse=True)
        return filtered_results
