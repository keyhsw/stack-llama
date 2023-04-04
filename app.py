import json
import os
from threading import Thread

import gradio as gr
import torch
from huggingface_hub import Repository
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, TextIteratorStreamer)

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)
# filesystem to save input and outputs
HF_TOKEN = os.environ.get("HF_TOKEN", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# if HF_TOKEN:
#     repo = Repository(
#         local_dir="data", clone_from="philschmid/playground-prompts", use_auth_token=HF_TOKEN, repo_type="dataset"
#     )


# Load peft config for pre-trained checkpoint etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceH4/llama-se-rl-ed"
if device == "cpu":
    model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
else:
    # torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16
    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

PROMPT_TEMPLATE = """Question: {prompt}\n\nAnswer: """


def generate(instruction, temperature, max_new_tokens, top_p, length_penalty):
    formatted_instruction = PROMPT_TEMPLATE.format(input=instruction)
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
    # if HF_TOKEN:
    #     save_inputs_and_outputs(formatted_instruction, output, generate_kwargs)
    return output


# def save_inputs_and_outputs(inputs, outputs, generate_kwargs):
#     with open(os.path.join("data", "prompts.jsonl"), "a") as f:
#         json.dump({"inputs": inputs, "outputs": outputs, "generate_kwargs": generate_kwargs}, f, ensure_ascii=False)
#         f.write("\n")
#         commit_url = repo.push_to_hub()


examples = [
    """Beantworten Sie die Frage am Ende des Textes anhand der folgenden Zusammenhänge. Wenn Sie die Antwort nicht wissen, sagen Sie, dass Sie es nicht wissen, versuchen Sie nicht, eine Antwort zu erfinden.
"Das Unternehmen wurde 2016 von den französischen Unternehmern Clément Delangue, Julien Chaumond und Thomas Wolf gegründet und entwickelte ursprünglich eine Chatbot-App, die sich an Teenager richtete.[2] Nachdem das Modell hinter dem Chatbot offengelegt wurde, konzentrierte sich das Unternehmen auf eine Plattform für maschinelles Lernen.

Im März 2021 sammelte Hugging Face in einer Serie-B-Finanzierungsrunde 40 Millionen US-Dollar ein[3].

Am 28. April 2021 rief das Unternehmen in Zusammenarbeit mit mehreren anderen Forschungsgruppen den BigScience Research Workshop ins Leben, um ein offenes großes Sprachmodell zu veröffentlichen.[4] Im Jahr 2022 wurde der Workshop mit der Ankündigung von BLOOM abgeschlossen, einem mehrsprachigen großen Sprachmodell mit 176 Milliarden Parametern.[5]"

Frage: Wann wurde Hugging Face gegründet?""",
    "Erklären Sie, was eine API ist.",
    "Bitte beantworten Sie die folgende Frage. Wer wird der nächste Ballon d'or sein?",
    "Beantworten Sie die folgende Ja/Nein-Frage, indem Sie Schritt für Schritt argumentieren. Kannst du ein ganzes Haiku in einem einzigen Tweet schreiben?",
    "Schreibe eine Produktbeschreibung für einen LG 43UQ75009LF 109 cm (43 Zoll) UHD Fernseher (Active HDR, 60 Hz, Smart TV) [Modelljahr 2022]",
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
                instruction = gr.Textbox(placeholder="Hier Anweisung eingeben...", label="Anweisung")
                output = gr.Textbox(
                    interactive=False,
                    lines=8,
                    label="Antwort",
                    placeholder="Hier Antwort erscheint...",
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
                    info="The higher more random",
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
                    label="Top p",
                    value=0.9,
                    minimum=0.0,
                    maximum=1,
                    step=0.05,
                    interactive=True,
                    info="probabilities that add up are kept",
                )
                length_penalty = gr.Slider(
                    label="Length penalty",
                    value=1.0,
                    minimum=-10.0,
                    maximum=10.0,
                    step=0.1,
                    interactive=True,
                    info="> 0.0 longer, < 0.0 shorter",
                )

    submit.click(generate, inputs=[instruction, temperature, max_new_tokens, top_p, length_penalty], outputs=[output])
    instruction.submit(
        generate, inputs=[instruction, temperature, max_new_tokens, top_p, length_penalty], outputs=[output]
    )

demo.queue()
demo.launch()
