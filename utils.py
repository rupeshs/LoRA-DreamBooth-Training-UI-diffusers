from __future__ import annotations

import pathlib


def find_exp_dirs(ignore_repo: bool = False) -> list[str]:
    repo_dir = pathlib.Path(__file__).parent
    exp_root_dir = repo_dir / 'experiments'
    if not exp_root_dir.exists():
        return []
    exp_dirs = sorted(exp_root_dir.glob('*'))
    exp_dirs = [
        exp_dir for exp_dir in exp_dirs
        if (exp_dir / 'pytorch_lora_weights.bin').exists()
    ]
    if ignore_repo:
        exp_dirs = [
            exp_dir for exp_dir in exp_dirs if not (exp_dir / '.git').exists()
        ]
    return [path.relative_to(repo_dir).as_posix() for path in exp_dirs]


def save_model_card(
    save_dir: pathlib.Path,
    base_model: str,
    instance_prompt: str,
    test_prompt: str = '',
    test_image_dir: str = '',
) -> None:
    image_str = ''
    if test_prompt and test_image_dir:
        image_paths = sorted((save_dir / test_image_dir).glob('*'))
        if image_paths:
            image_str = f'Test prompt: {test_prompt}\n'
            for image_path in image_paths:
                rel_path = image_path.relative_to(save_dir)
                image_str += f'![{image_path.stem}]({rel_path})\n'

    model_card = f'''---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {instance_prompt}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
# LoRA DreamBooth - {save_dir.name}

These are LoRA adaption weights for [{base_model}](https://huggingface.co/{base_model}). The weights were trained on the instance prompt "{instance_prompt}" using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following.

{image_str}
'''

    with open(save_dir / 'README.md', 'w') as f:
        f.write(model_card)
