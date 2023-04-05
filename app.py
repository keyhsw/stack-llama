import os
from threading import Thread

import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "trl-lib/llama-se-rl-merged"
print(f"Loading model: {model_id}")
if device == "cpu":
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, use_auth_token=HF_TOKEN)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", load_in_8bit=True, use_auth_token=HF_TOKEN
    )

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)

PROMPT_TEMPLATE = """Question: {prompt}\n\nAnswer:"""


def generate(instruction, temperature=0.7, max_new_tokens=256, top_p=0.95, top_k=40):
    formatted_instruction = PROMPT_TEMPLATE.format(prompt=instruction)

    temperature = float(temperature)
    top_p = float(top_p)
    streamer = TextIteratorStreamer(tokenizer)
    model_inputs = tokenizer(formatted_instruction, return_tensors="pt", truncation=True, max_length=2048).to(device)

    generate_kwargs = dict(
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    t = Thread(target=model.generate, kwargs={**dict(model_inputs, streamer=streamer), **generate_kwargs})
    t.start()

    output = ""
    hidden_output = ""
    for new_text in streamer:
        # skip streaming until new text is available
        if len(hidden_output) <= len(formatted_instruction):
            hidden_output += new_text
            continue
        # replace eos token
        # if tokenizer.eos_token in new_text:
        #     new_text = new_text.replace(tokenizer.eos_token, "")
        output += new_text
        yield output
    return output


examples = [
    "How do I create an array in C++ of length 5 which contains all even numbers between 1 and 10?",
    "How can I write a Java function to generate the nth Fibonacci number?",
    "How can I sort a list in Python?",
]


def process_example(args):
    for x in generate(args):
        pass
    return x


with gr.Blocks(theme=theme, analytics_enabled=False) as demo:
    with gr.Column():
        gr.Markdown(
            """<h1><center>🦙🦙🦙 StackLLaMa 🦙🦙🦙</center></h1>

            StackLLaMa is a 7 billion parameter language model that has been trained on pairs of programming questions and answers from [Stack Overflow](https://stackoverflow.com) using Reinforcement Learning from Human Feedback with the [TRL library](https://github.com/lvwerra/trl). For more details, check out our blog post [ADD LINK].

            Type in the box below and click the button to generate answers to your most pressing coding questions 🔥!
      """
        )
        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question")
                with gr.Box():
                    gr.Markdown("**Answer**")
                    output = gr.Markdown()
                # output = gr.Textbox(
                #     interactive=False,
                #     lines=8,
                #     label="Answer",
                #     placeholder="Here will be the answer to your question",
                # )
                submit = gr.Button("Generate", variant="primary")
                gr.Examples(
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=True,
                    fn=process_example,
                    outputs=[output],
                )

            with gr.Column(scale=1):
                temperature = gr.Slider(
                    label="Temperature",
                    value=0.7,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    interactive=True,
                    info="Higher values produce more diverse outputs",
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=64,
                    minimum=0,
                    maximum=2048,
                    step=4,
                    interactive=True,
                    info="The maximum numbers of new tokens",
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    value=0.95,
                    minimum=0.0,
                    maximum=1,
                    step=0.05,
                    interactive=True,
                    info="Higher values sample more low-probability tokens",
                )
                top_k = gr.Slider(
                    label="Top-k",
                    value=40,
                    minimum=0,
                    maximum=100,
                    step=2,
                    interactive=True,
                    info="Sample from top-k tokens",
                )

    submit.click(generate, inputs=[instruction, temperature, max_new_tokens, top_p, top_k], outputs=[output])
    instruction.submit(generate, inputs=[instruction, temperature, max_new_tokens, top_p, top_k], outputs=[output])

demo.queue(concurrency_count=1)
demo.launch(enable_queue=True, share=True)
