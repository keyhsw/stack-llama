import os
from threading import Thread

import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, TextIteratorStreamer)

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load peft config for pre-trained checkpoint etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceH4/llama-se-rl-ed"
if device == "cpu":
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, use_auth_token=HF_TOKEN)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", load_in_8bit=True, use_auth_token=HF_TOKEN
    )

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)

PROMPT_TEMPLATE = """Question: {prompt}\n\nAnswer: """


def generate(instruction, temperature, max_new_tokens, top_p, length_penalty):
    formatted_instruction = PROMPT_TEMPLATE.format(prompt=instruction)
    # COMMENT IN FOR NON STREAMING
    # generation_config = GenerationConfig(
    #     do_sample=True,
    #     top_p=top_p,
    #     temperature=temperature,
    #     max_new_tokens=max_new_tokens,
    #     early_stopping=True,
    #     length_penalty=length_penalty,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.pad_token_id,
    # )

    # input_ids = tokenizer(
    #     formatted_instruction, return_tensors="pt", truncation=True, max_length=2048
    # ).input_ids.cuda()

    # with torch.inference_mode(), torch.autocast("cuda"):
    #     outputs = model.generate(input_ids=input_ids, generation_config=generation_config)[0]

    # output = tokenizer.decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    # return output.split("### Antwort:\n")[1]

    # STREAMING BASED ON git+https://github.com/gante/transformers.git@streamer_iterator

    # streaming
    streamer = TextIteratorStreamer(tokenizer)
    model_inputs = tokenizer(formatted_instruction, return_tensors="pt", truncation=True, max_length=2048).to(device)

    generate_kwargs = dict(
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
        length_penalty=length_penalty,
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
        if tokenizer.eos_token in new_text:
            new_text = new_text.replace(tokenizer.eos_token, "")
        output += new_text
        yield output
    return output


examples = [
    "How do I create an array in C++ of length 5 which contains all even numbers between 1 and 10?",
    "How can I write a Java function to generate the nth Fibonacci number?",
    "How can I write a Python function that checks if a given number is a palindrome or not?",
    "What is the output of the following code?\n\n```\nlist1 = ['a', 'b', 'c']\nlist2 = [1, 2, 3]\n\nfor x, y in zip(list1, list2):\n    print(x * y)\n```",
]


with gr.Blocks(theme=theme) as demo:
    with gr.Column():
        gr.Markdown(
            """<h1><center>🦙🦙🦙 StackLLaMa 🦙🦙🦙</center></h1>

            StackLLaMa is a 7 billion parameter language model that has been trained on pairs of programming questions and answers from [Stack Overflow](https://stackoverflow.com) using Reinforcement Learning from Human Feedback (RLHF) with the [TRL library](https://github.com/lvwerra/trl). For more details, check out our blog post [ADD LINK].

            Type in the box below and click the button to generate answers to your most pressing coding questions 🔥!
      """
        )
        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question")
                output = gr.Textbox(
                    interactive=False,
                    lines=8,
                    label="Answer",
                    placeholder="Here will be the answer to your question",
                )
                submit = gr.Button("Generate", variant="primary")
                gr.Examples(examples=examples, inputs=[instruction])

            with gr.Column(scale=1):
                temperature = gr.Slider(
                    label="Temperature",
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    interactive=True,
                    info="Higher values produce more diverse outputs",
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=256,
                    minimum=0,
                    maximum=2048,
                    step=5,
                    interactive=True,
                    info="The maximum numbers of new tokens",
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    value=0.9,
                    minimum=0.0,
                    maximum=1,
                    step=0.05,
                    interactive=True,
                    info="Higher values sample fewer low-probability tokens",
                )
                length_penalty = gr.Slider(
                    label="Length penalty",
                    value=1.0,
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.1,
                    interactive=True,
                    info="> 0 longer, < 0 shorter",
                )

    submit.click(generate, inputs=[instruction, temperature, max_new_tokens, top_p, length_penalty], outputs=[output])
    instruction.submit(
        generate, inputs=[instruction, temperature, max_new_tokens, top_p, length_penalty], outputs=[output]
    )

demo.queue()
demo.launch()
