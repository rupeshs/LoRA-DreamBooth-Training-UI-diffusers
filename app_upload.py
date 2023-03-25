#!/usr/bin/env python

from __future__ import annotations

import pathlib

import gradio as gr
import slugify

from constants import UploadTarget
from uploader import Uploader
from utils import find_exp_dirs


class LoRAModelUploader(Uploader):
    def upload_lora_model(
        self,
        folder_path: str,
        repo_name: str,
        upload_to: str,
        private: bool,
        delete_existing_repo: bool,
    ) -> str:
        if not folder_path:
            raise ValueError
        if not repo_name:
            repo_name = pathlib.Path(folder_path).name
        repo_name = slugify.slugify(repo_name)

        if upload_to == UploadTarget.PERSONAL_PROFILE.value:
            organization = ''
        elif upload_to == UploadTarget.LORA_LIBRARY.value:
            organization = 'lora-library'
        else:
            raise ValueError

        return self.upload(folder_path,
                           repo_name,
                           organization=organization,
                           private=private,
                           delete_existing_repo=delete_existing_repo)


def load_local_lora_model_list() -> dict:
    choices = find_exp_dirs(ignore_repo=True)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def create_upload_demo(hf_token: str | None) -> gr.Blocks:
    uploader = LoRAModelUploader(hf_token)
    model_dirs = find_exp_dirs(ignore_repo=True)

    with gr.Blocks() as demo:
        with gr.Box():
            gr.Markdown('Local Models')
            reload_button = gr.Button('Reload Model List')
            model_dir = gr.Dropdown(
                label='Model names',
                choices=model_dirs,
                value=model_dirs[0] if model_dirs else None)
        with gr.Box():
            gr.Markdown('Upload Settings')
            with gr.Row():
                use_private_repo = gr.Checkbox(label='Private', value=True)
                delete_existing_repo = gr.Checkbox(
                    label='Delete existing repo of the same name', value=False)
            upload_to = gr.Radio(label='Upload to',
                                 choices=[_.value for _ in UploadTarget],
                                 value=UploadTarget.LORA_LIBRARY.value)
            model_name = gr.Textbox(label='Model Name')
        upload_button = gr.Button('Upload')
        gr.Markdown('''
            - You can upload your trained model to your personal profile (i.e. https://huggingface.co/{your_username}/{model_name}) or to the public [LoRA Concepts Library](https://huggingface.co/lora-library) (i.e. https://huggingface.co/lora-library/{model_name}).
            ''')
        with gr.Box():
            gr.Markdown('Output message')
            output_message = gr.Markdown()

        reload_button.click(fn=load_local_lora_model_list,
                            inputs=None,
                            outputs=model_dir)
        upload_button.click(fn=uploader.upload_lora_model,
                            inputs=[
                                model_dir,
                                model_name,
                                upload_to,
                                use_private_repo,
                                delete_existing_repo,
                            ],
                            outputs=output_message)

    return demo


if __name__ == '__main__':
    import os

    hf_token = os.getenv('HF_TOKEN')
    demo = create_upload_demo(hf_token)
    demo.queue(max_size=1).launch(share=False)
