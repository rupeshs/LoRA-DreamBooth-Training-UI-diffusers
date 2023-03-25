#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr

from constants import UploadTarget
from inference import InferencePipeline
from trainer import Trainer


def create_training_demo(trainer: Trainer,
                         pipe: InferencePipeline | None = None) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown('Training Data')
                    instance_images = gr.Files(label='Instance images')
                    instance_prompt = gr.Textbox(label='Instance prompt',
                                                 max_lines=1)
                    gr.Markdown('''
                        - Upload images of the style you are planning on training on.
                        - For an instance prompt, use a unique, made up word to avoid collisions.
                        ''')
                with gr.Box():
                    gr.Markdown('Output Model')
                    output_model_name = gr.Text(label='Name of your model',
                                                max_lines=1)
                    delete_existing_model = gr.Checkbox(
                        label='Delete existing model of the same name',
                        value=False)
                    validation_prompt = gr.Text(label='Validation Prompt')
                with gr.Box():
                    gr.Markdown('Upload Settings')
                    with gr.Row():
                        upload_to_hub = gr.Checkbox(
                            label='Upload model to Hub', value=True)
                        use_private_repo = gr.Checkbox(label='Private',
                                                       value=True)
                        delete_existing_repo = gr.Checkbox(
                            label='Delete existing repo of the same name',
                            value=False)
                    upload_to = gr.Radio(
                        label='Upload to',
                        choices=[_.value for _ in UploadTarget],
                        value=UploadTarget.LORA_LIBRARY.value)
                    gr.Markdown('''
                    - By default, trained models will be uploaded to [LoRA Library](https://huggingface.co/lora-library) (see [this example model](https://huggingface.co/lora-library/lora-dreambooth-sample-dog)).
                    - You can also choose "Personal Profile", in which case, the model will be uploaded to https://huggingface.co/{your_username}/{model_name}.
                    ''')

            with gr.Box():
                gr.Markdown('Training Parameters')
                with gr.Row():
                    base_model = gr.Text(
                        label='Base Model',
                        value='stabilityai/stable-diffusion-2-1-base',
                        max_lines=1)
                    resolution = gr.Dropdown(choices=['512', '768'],
                                             value='512',
                                             label='Resolution')
                num_training_steps = gr.Number(
                    label='Number of Training Steps', value=1000, precision=0)
                learning_rate = gr.Number(label='Learning Rate', value=0.0001)
                gradient_accumulation = gr.Number(
                    label='Number of Gradient Accumulation',
                    value=1,
                    precision=0)
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=0)
                fp16 = gr.Checkbox(label='FP16', value=True)
                use_8bit_adam = gr.Checkbox(label='Use 8bit Adam', value=True)
                checkpointing_steps = gr.Number(label='Checkpointing Steps',
                                                value=100,
                                                precision=0)
                use_wandb = gr.Checkbox(label='Use W&B',
                                        value=False,
                                        interactive=bool(
                                            os.getenv('WANDB_API_KEY')))
                validation_epochs = gr.Number(label='Validation Epochs',
                                              value=100,
                                              precision=0)
                gr.Markdown('''
                    - The base model must be a model that is compatible with [diffusers](https://github.com/huggingface/diffusers) library.
                    - It takes a few minutes to download the base model first.
                    - It will take about 8 minutes to train for 1000 steps with a T4 GPU.
                    - You may want to try a small number of steps first, like 1, to see if everything works fine in your environment.
                    - You can check the training status by pressing the "Open logs" button if you are running this on your Space.
                    - You need to set the environment variable `WANDB_API_KEY` if you'd like to use [W&B](https://wandb.ai/site). See [W&B documentation](https://docs.wandb.ai/guides/track/advanced/environment-variables).
                    - **Note:** Due to [this issue](https://github.com/huggingface/accelerate/issues/944), currently, training will not terminate properly if you use W&B.
                    ''')

        remove_gpu_after_training = gr.Checkbox(
            label='Remove GPU after training',
            value=False,
            interactive=bool(os.getenv('SPACE_ID')),
            visible=False)
        run_button = gr.Button('Start Training')

        with gr.Box():
            gr.Markdown('Output message')
            output_message = gr.Markdown()

        if pipe is not None:
            run_button.click(fn=pipe.clear)
        run_button.click(fn=trainer.run,
                         inputs=[
                             instance_images,
                             instance_prompt,
                             output_model_name,
                             delete_existing_model,
                             validation_prompt,
                             base_model,
                             resolution,
                             num_training_steps,
                             learning_rate,
                             gradient_accumulation,
                             seed,
                             fp16,
                             use_8bit_adam,
                             checkpointing_steps,
                             use_wandb,
                             validation_epochs,
                             upload_to_hub,
                             use_private_repo,
                             delete_existing_repo,
                             upload_to,
                             remove_gpu_after_training,
                         ],
                         outputs=output_message)
    return demo


if __name__ == '__main__':
    hf_token = os.getenv('HF_TOKEN')
    trainer = Trainer(hf_token)
    demo = create_training_demo(trainer)
    demo.queue(max_size=1).launch(share=False)
