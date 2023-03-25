from __future__ import annotations

import datetime
import os
import pathlib
import shlex
import shutil
import subprocess

import gradio as gr
import PIL.Image
import slugify
import torch
from huggingface_hub import HfApi

from app_upload import LoRAModelUploader
from utils import save_model_card

URL_TO_JOIN_LORA_LIBRARY_ORG = 'https://huggingface.co/organizations/lora-library/share/hjetHAcKjnPHXhHfbeEcqnBqmhgilFfpOL'


def pad_image(image: PIL.Image.Image) -> PIL.Image.Image:
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = PIL.Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = PIL.Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image


class Trainer:
    def __init__(self, hf_token: str | None = None):
        self.hf_token = hf_token
        self.api = HfApi(token=hf_token)
        self.model_uploader = LoRAModelUploader(hf_token)

    def prepare_dataset(self, instance_images: list, resolution: int,
                        instance_data_dir: pathlib.Path) -> None:
        shutil.rmtree(instance_data_dir, ignore_errors=True)
        instance_data_dir.mkdir(parents=True)
        for i, temp_path in enumerate(instance_images):
            image = PIL.Image.open(temp_path.name)
            image = pad_image(image)
            image = image.resize((resolution, resolution))
            image = image.convert('RGB')
            out_path = instance_data_dir / f'{i:03d}.jpg'
            image.save(out_path, format='JPEG', quality=100)

    def join_lora_library_org(self) -> None:
        subprocess.run(
            shlex.split(
                f'curl -X POST -H "Authorization: Bearer {self.hf_token}" -H "Content-Type: application/json" {URL_TO_JOIN_LORA_LIBRARY_ORG}'
            ))

    def run(
        self,
        instance_images: list | None,
        instance_prompt: str,
        output_model_name: str,
        overwrite_existing_model: bool,
        validation_prompt: str,
        base_model: str,
        resolution_s: str,
        n_steps: int,
        learning_rate: float,
        gradient_accumulation: int,
        seed: int,
        fp16: bool,
        use_8bit_adam: bool,
        checkpointing_steps: int,
        use_wandb: bool,
        validation_epochs: int,
        upload_to_hub: bool,
        use_private_repo: bool,
        delete_existing_repo: bool,
        upload_to: str,
        remove_gpu_after_training: bool,
    ) -> str:
        if not torch.cuda.is_available():
            raise gr.Error('CUDA is not available.')
        if instance_images is None:
            raise gr.Error('You need to upload images.')
        if not instance_prompt:
            raise gr.Error('The instance prompt is missing.')
        if not validation_prompt:
            raise gr.Error('The validation prompt is missing.')

        resolution = int(resolution_s)

        if not output_model_name:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            output_model_name = f'lora-dreambooth-{timestamp}'
        output_model_name = slugify.slugify(output_model_name)

        repo_dir = pathlib.Path(__file__).parent
        output_dir = repo_dir / 'experiments' / output_model_name
        if overwrite_existing_model or upload_to_hub:
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)

        instance_data_dir = repo_dir / 'training_data' / output_model_name
        self.prepare_dataset(instance_images, resolution, instance_data_dir)

        if upload_to_hub:
            self.join_lora_library_org()

        command = f'''
        accelerate launch train_dreambooth_lora.py \
          --pretrained_model_name_or_path={base_model}  \
          --instance_data_dir={instance_data_dir} \
          --output_dir={output_dir} \
          --instance_prompt="{instance_prompt}" \
          --resolution={resolution} \
          --train_batch_size=1 \
          --gradient_accumulation_steps={gradient_accumulation} \
          --learning_rate={learning_rate} \
          --lr_scheduler=constant \
          --lr_warmup_steps=0 \
          --max_train_steps={n_steps} \
          --checkpointing_steps={checkpointing_steps} \
          --validation_prompt="{validation_prompt}" \
          --validation_epochs={validation_epochs} \
          --seed={seed}
        '''
        if fp16:
            command += ' --mixed_precision fp16'
        if use_8bit_adam:
            command += ' --use_8bit_adam'
        if use_wandb:
            command += ' --report_to wandb'

        with open(output_dir / 'train.sh', 'w') as f:
            command_s = ' '.join(command.split())
            f.write(command_s)
        subprocess.run(shlex.split(command))
        save_model_card(save_dir=output_dir,
                        base_model=base_model,
                        instance_prompt=instance_prompt,
                        test_prompt=validation_prompt,
                        test_image_dir='test_images')

        message = 'Training completed!'
        print(message)

        if upload_to_hub:
            upload_message = self.model_uploader.upload_lora_model(
                folder_path=output_dir.as_posix(),
                repo_name=output_model_name,
                upload_to=upload_to,
                private=use_private_repo,
                delete_existing_repo=delete_existing_repo)
            print(upload_message)
            message = message + '\n' + upload_message

        if remove_gpu_after_training:
            space_id = os.getenv('SPACE_ID')
            if space_id:
                self.api.request_space_hardware(repo_id=space_id,
                                                hardware='cpu-basic')

        return message
