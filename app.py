import json
import os

import gradio as gr
from huggingface_hub import Repository
from text_generation import Client

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_URL = os.environ.get("API_URL")


theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)
if HF_TOKEN:
    repo = Repository(
        local_dir="data", clone_from="trl-lib/stack-llama-prompts", use_auth_token=HF_TOKEN, repo_type="dataset"
    )

client = Client(
    API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)

PROMPT_TEMPLATE = """Question: {prompt}\n\nAnswer:"""


def save_inputs_and_outputs(inputs, outputs, generate_kwargs):
    with open(os.path.join("data", "prompts.jsonl"), "a") as f:
        json.dump({"inputs": inputs, "outputs": outputs, "generate_kwargs": generate_kwargs}, f, ensure_ascii=False)
        f.write("\n")
        commit_url = repo.push_to_hub()


# def generate(instruction, temperature=0.9, max_new_tokens=128, top_p=0.95, top_k=100):
#     set_seed(42)
#     formatted_instruction = PROMPT_TEMPLATE.format(prompt=instruction)

#     temperature = float(temperature)
#     top_p = float(top_p)
#     streamer = TextIteratorStreamer(tokenizer)
#     model_inputs = tokenizer(formatted_instruction, return_tensors="pt", truncation=True, max_length=2048).to(device)

#     generate_kwargs = dict(
#         top_p=top_p,
#         temperature=temperature,
#         max_new_tokens=max_new_tokens,
#         do_sample=True,
#         top_k=top_k,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.eos_token_id,
#     )
#     t = Thread(target=model.generate, kwargs={**dict(model_inputs, streamer=streamer), **generate_kwargs})
#     t.start()

#     output = ""
#     hidden_output = ""
#     for new_text in streamer:
#         # skip streaming until new text is available
#         if len(hidden_output) <= len(formatted_instruction):
#             hidden_output += new_text
#             continue
#         # replace eos token
#         # if tokenizer.eos_token in new_text:
#         #     new_text = new_text.replace(tokenizer.eos_token, "")
#         output += new_text
#         yield output
#     if HF_TOKEN:
#         print("Pushing prompt and completion to the Hub")
#         save_inputs_and_outputs(formatted_instruction, output, generate_kwargs)
#     return output


def generate(instruction, temperature=0.9, max_new_tokens=256, top_p=0.95, top_k=100):
    formatted_instruction = PROMPT_TEMPLATE.format(prompt=instruction)

    temperature = float(temperature)
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        truncate=999,
        seed=42,
    )

    stream = client.generate_stream(
        formatted_instruction,
        **generate_kwargs,
    )

    output = ""
    for response in stream:
        output += response.token.text
        yield output
    if HF_TOKEN:
        print("Pushing prompt and completion to the Hub")
        save_inputs_and_outputs(formatted_instruction, output, generate_kwargs)

    return output

    # streamer = TextIteratorStreamer(tokenizer)
    # model_inputs = tokenizer(formatted_instruction, return_tensors="pt", truncation=True, max_length=2048).to(device)

    # generate_kwargs = dict(
    #     top_p=top_p,
    #     temperature=temperature,
    #     max_new_tokens=max_new_tokens,
    #     do_sample=True,
    #     top_k=top_k,
    #     # eos_token_id=tokenizer.eos_token_id,
    #     # pad_token_id=tokenizer.eos_token_id,
    # )
    # t = Thread(target=model.generate, kwargs={**dict(model_inputs, streamer=streamer), **generate_kwargs})
    # t.start()

    # output = ""
    # hidden_output = ""
    # for new_text in streamer:
    #     # skip streaming until new text is available
    #     if len(hidden_output) <= len(formatted_instruction):
    #         hidden_output += new_text
    #         continue
    #     # replace eos token
    #     # if tokenizer.eos_token in new_text:
    #     #     new_text = new_text.replace(tokenizer.eos_token, "")
    #     output += new_text
    #     yield output
    if HF_TOKEN:
        print("Pushing prompt and completion to the Hub")
        save_inputs_and_outputs(formatted_instruction, output, generate_kwargs)
    # return output


examples = [
    "A llama is in my lawn. How do I get rid of him?",
    "How do I create an array in C++ which contains all even numbers between 1 and 10?",
    "How can I sort a list in Python?",
    "How can I write a Java function to generate the nth Fibonacci number?",
    "How many helicopters can a llama eat in one sitting?",
]


def process_example(args):
    for x in generate(args):
        pass
    return x


with gr.Blocks(theme=theme, analytics_enabled=False, css=".generating {visibility: hidden}") as demo:
    with gr.Column():
        gr.Markdown(
            """<h1><center>🦙🦙🦙 StackLLaMa 🦙🦙🦙</center></h1>

            StackLLaMa is a 7 billion parameter language model that has been trained on pairs of questions and answers from [Stack Exchange](https://stackexchange.com) using Reinforcement Learning from Human Feedback with the [TRL library](https://github.com/lvwerra/trl). For more details, check out our [blog post](https://huggingface.co/blog/stackllama).

            Type in the box below and click the button to generate answers to your most pressing questions 🔥!

            **Note:** we are collecting your prompts and model completions for research purposes.
      """
        )
        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question")
                with gr.Box():
                    gr.Markdown("**Answer**")
                    output = gr.Markdown()
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
                    value=0.8,
                    minimum=0.01,
                    maximum=2.0,
                    step=0.1,
                    interactive=True,
                    info="Higher values produce more diverse outputs",
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=128,
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
demo.launch(enable_queue=True)  # , share=True)
