import json
import os
import shutil

import gradio as gr
from huggingface_hub import Repository
from text_generation import Client

from share_btn import community_icon_html, loading_icon_html, share_js, share_btn_css

HF_TOKEN = os.environ.get("TRL_TOKEN", None)
API_URL = os.environ.get("API_URL")


theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)
if HF_TOKEN:
    try:
        shutil.rmtree("./data/")
    except:
        pass

    repo = Repository(
        local_dir="./data/", clone_from="trl-lib/stack-llama-prompts", use_auth_token=HF_TOKEN, repo_type="dataset"
    )
    repo.git_pull()

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


def generate(instruction, temperature=0.9, max_new_tokens=256, top_p=0.95, top_k=100, do_save=True):
    formatted_instruction = PROMPT_TEMPLATE.format(prompt=instruction)

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,
        truncate=999,
        seed=42,
        stop_sequences=["</s>"],
    )

    stream = client.generate_stream(
        formatted_instruction,
        **generate_kwargs,
    )

    output = ""
    for response in stream:
        output += response.token.text
        yield output
    if HF_TOKEN and do_save:
        try:
            print("Pushing prompt and completion to the Hub")
            save_inputs_and_outputs(formatted_instruction, output, generate_kwargs)
        except Exception,e:
            print(e)
            
    return output


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

css = ".generating {visibility: hidden}" + share_btn_css

with gr.Blocks(theme=theme, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(
            """![](https://huggingface.co/spaces/trl-lib/stack-llama/resolve/main/stackllama_logo.png)


            StackLLaMa is a 7 billion parameter language model that has been trained on pairs of questions and answers from [Stack Exchange](https://stackexchange.com) using Reinforcement Learning from Human Feedback with the [TRL library](https://github.com/lvwerra/trl). For more details, check out our [blog post](https://huggingface.co/blog/stackllama).

            Type in the box below and click the button to generate answers to your most pressing questions!
      """
        )
        do_save = gr.Checkbox(value=True, label="You consent to the storage of your prompt and generated text for research and development purposes.")
        with gr.Row():
            with gr.Column(scale=3):
                instruction = gr.Textbox(placeholder="Enter your question here", label="Question", elem_id="q-input")
                with gr.Box():
                    gr.Markdown("**Answer**")
                    output = gr.Markdown(elem_id="q-output")
                submit = gr.Button("Generate", variant="primary")
                with gr.Group(elem_id="share-btn-container"):
                    community_icon = gr.HTML(community_icon_html, visible=True)
                    loading_icon = gr.HTML(loading_icon_html, visible=True)
                    share_button = gr.Button("Share to community", elem_id="share-btn", visible=True)
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
                    value=0.9,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    interactive=True,
                    info="Higher values produce more diverse outputs",
                )
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    value=128,
                    minimum=0,
                    maximum=512,
                    step=4,
                    interactive=True,
                    info="The maximum numbers of new tokens",
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    value=0.90,
                    minimum=0.0,
                    maximum=1,
                    step=0.05,
                    interactive=True,
                    info="Higher values sample more low-probability tokens",
                )
                top_k = gr.Slider(
                    label="Top-k",
                    value=50,
                    minimum=0,
                    maximum=100,
                    step=2,
                    interactive=True,
                    info="Sample from top-k tokens",
                )

    submit.click(generate, inputs=[instruction, temperature, max_new_tokens, top_p, top_k, do_save], outputs=[output])
    instruction.submit(generate, inputs=[instruction, temperature, max_new_tokens, top_p, top_k], outputs=[output])
    share_button.click(None, [], [], _js=share_js)

demo.queue(concurrency_count=16).launch(debug=True)