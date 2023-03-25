from __future__ import annotations

import gc
import pathlib

import gradio as gr
import PIL.Image
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import ModelCard


class InferencePipeline:
    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token
        self.pipe = None
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.lora_model_id = None
        self.base_model_id = None

    def clear(self) -> None:
        self.lora_model_id = None
        self.base_model_id = None
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def check_if_model_is_local(lora_model_id: str) -> bool:
        return pathlib.Path(lora_model_id).exists()

    @staticmethod
    def get_model_card(model_id: str,
                       hf_token: str | None = None) -> ModelCard:
        if InferencePipeline.check_if_model_is_local(model_id):
            card_path = (pathlib.Path(model_id) / 'README.md').as_posix()
        else:
            card_path = model_id
        return ModelCard.load(card_path, token=hf_token)

    @staticmethod
    def get_base_model_info(lora_model_id: str,
                            hf_token: str | None = None) -> str:
        card = InferencePipeline.get_model_card(lora_model_id, hf_token)
        return card.data.base_model

    def load_pipe(self, lora_model_id: str) -> None:
        if lora_model_id == self.lora_model_id:
            return
        base_model_id = self.get_base_model_info(lora_model_id, self.hf_token)
        if base_model_id != self.base_model_id:
            if self.device.type == 'cpu':
                pipe = DiffusionPipeline.from_pretrained(
                    base_model_id, use_auth_token=self.hf_token)
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float16,
                    use_auth_token=self.hf_token)
                pipe = pipe.to(self.device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config)
            self.pipe = pipe
        self.pipe.unet.load_attn_procs(  # type: ignore
            lora_model_id, use_auth_token=self.hf_token)

        self.lora_model_id = lora_model_id  # type: ignore
        self.base_model_id = base_model_id  # type: ignore

    def run(
        self,
        lora_model_id: str,
        prompt: str,
        lora_scale: float,
        seed: int,
        n_steps: int,
        guidance_scale: float,
    ) -> PIL.Image.Image:
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')

        self.load_pipe(lora_model_id)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        out = self.pipe(
            prompt,
            num_inference_steps=n_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            cross_attention_kwargs={'scale': lora_scale},
        )  # type: ignore
        return out.images[0]
