#!/usr/bin/env python

from __future__ import annotations

import enum

import gradio as gr
from huggingface_hub import HfApi

from inference import InferencePipeline
from utils import find_exp_dirs

SAMPLE_MODEL_IDS = [
    'patrickvonplaten/lora_dreambooth_dog_example',
    'sayakpaul/sd-model-finetuned-lora-t4',
]


class ModelSource(enum.Enum):
    SAMPLE = 'Sample'
    HUB_LIB = 'Hub (lora-library)'
    LOCAL = 'Local'


class InferenceUtil:
    def __init__(self, hf_token: str | None):
        self.hf_token = hf_token

    @staticmethod
    def load_sample_lora_model_list():
        return gr.update(choices=SAMPLE_MODEL_IDS, value=SAMPLE_MODEL_IDS[0])

    def load_hub_lora_model_list(self) -> dict:
        api = HfApi(token=self.hf_token)
        choices = [
            info.modelId for info in api.list_models(author='lora-library')
        ]
        return gr.update(choices=choices,
                         value=choices[0] if choices else None)

    @staticmethod
    def load_local_lora_model_list() -> dict:
        choices = find_exp_dirs()
        return gr.update(choices=choices,
                         value=choices[0] if choices else None)

    def reload_lora_model_list(self, model_source: str) -> dict:
        if model_source == ModelSource.SAMPLE.value:
            return self.load_sample_lora_model_list()
        elif model_source == ModelSource.HUB_LIB.value:
            return self.load_hub_lora_model_list()
        elif model_source == ModelSource.LOCAL.value:
            return self.load_local_lora_model_list()
        else:
            raise ValueError

    def load_model_info(self, lora_model_id: str) -> tuple[str, str]:
        try:
            card = InferencePipeline.get_model_card(lora_model_id,
                                                    self.hf_token)
        except Exception:
            return '', ''
        base_model = getattr(card.data, 'base_model', '')
        instance_prompt = getattr(card.data, 'instance_prompt', '')
        return base_model, instance_prompt

    def reload_lora_model_list_and_update_model_info(
            self, model_source: str) -> tuple[dict, str, str]:
        model_list_update = self.reload_lora_model_list(model_source)
        model_list = model_list_update['choices']
        model_info = self.load_model_info(model_list[0] if model_list else '')
        return model_list_update, *model_info


def create_inference_demo(pipe: InferencePipeline,
                          hf_token: str | None = None) -> gr.Blocks:
    app = InferenceUtil(hf_token)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    model_source = gr.Radio(
                        label='Model Source',
                        choices=[_.value for _ in ModelSource],
                        value=ModelSource.SAMPLE.value)
                    reload_button = gr.Button('Reload Model List')
                    lora_model_id = gr.Dropdown(label='LoRA Model ID',
                                                choices=SAMPLE_MODEL_IDS,
                                                value=SAMPLE_MODEL_IDS[0])
                    with gr.Accordion(
                            label=
                            'Model info (Base model and instance prompt used for training)',
                            open=False):
                        with gr.Row():
                            base_model_used_for_training = gr.Text(
                                label='Base model', interactive=False)
                            instance_prompt_used_for_training = gr.Text(
                                label='Instance prompt', interactive=False)
                prompt = gr.Textbox(
                    label='Prompt',
                    max_lines=1,
                    placeholder='Example: "A picture of a sks dog in a bucket"'
                )
                alpha = gr.Slider(label='LoRA alpha',
                                  minimum=0,
                                  maximum=2,
                                  step=0.05,
                                  value=1)
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=0)
                with gr.Accordion('Other Parameters', open=False):
                    num_steps = gr.Slider(label='Number of Steps',
                                          minimum=0,
                                          maximum=100,
                                          step=1,
                                          value=25)
                    guidance_scale = gr.Slider(label='CFG Scale',
                                               minimum=0,
                                               maximum=50,
                                               step=0.1,
                                               value=7.5)

                run_button = gr.Button('Generate')

                gr.Markdown('''
                - After training, you can press "Reload Model List" button to load your trained model names.
                ''')
            with gr.Column():
                result = gr.Image(label='Result')

        model_source.change(
            fn=app.reload_lora_model_list_and_update_model_info,
            inputs=model_source,
            outputs=[
                lora_model_id,
                base_model_used_for_training,
                instance_prompt_used_for_training,
            ])
        reload_button.click(
            fn=app.reload_lora_model_list_and_update_model_info,
            inputs=model_source,
            outputs=[
                lora_model_id,
                base_model_used_for_training,
                instance_prompt_used_for_training,
            ])
        lora_model_id.change(fn=app.load_model_info,
                             inputs=lora_model_id,
                             outputs=[
                                 base_model_used_for_training,
                                 instance_prompt_used_for_training,
                             ])
        inputs = [
            lora_model_id,
            prompt,
            alpha,
            seed,
            num_steps,
            guidance_scale,
        ]
        prompt.submit(fn=pipe.run, inputs=inputs, outputs=result)
        run_button.click(fn=pipe.run, inputs=inputs, outputs=result)
    return demo


if __name__ == '__main__':
    import os

    hf_token = os.getenv('HF_TOKEN')
    pipe = InferencePipeline(hf_token)
    demo = create_inference_demo(pipe, hf_token)
    demo.queue(max_size=10).launch(share=False)
