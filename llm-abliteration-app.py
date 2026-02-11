import os
import random
import torch
import gradio as gr

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, login


torch.set_grad_enabled(False)

# ---------------- CONFIG ----------------

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
INSTRUCTION_COUNT = 32
DEFAULT_LAYER_RATIO = 0.6
POS = -1

HARMFUL_FILE = "harmful.txt"
HARMLESS_FILE = "harmless.txt"


# ---------------- MODEL LOADING ----------------

def load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------- DATA ----------------

def read_instruction_files():
    with open(HARMFUL_FILE, "r") as f:
        harmful = f.readlines()

    with open(HARMLESS_FILE, "r") as f:
        harmless = f.readlines()

    return harmful, harmless


# ---------------- CORE LOGIC ----------------

def generate_hidden_states(model, tokenizer, instructions, layer_idx):
    hidden_states = []

    for insn in tqdm(instructions, desc="Generating hidden states"):
        enc = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": insn}],
            add_generation_prompt=True,
            return_tensors="pt"
        )

        attention_mask = torch.ones_like(enc)

        out = model.generate(
            enc.to(model.device),
            attention_mask=attention_mask.to(model.device),
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        h = out.hidden_states[0][layer_idx][:, POS, :]
        hidden_states.append(h)

    return torch.stack(hidden_states)


def compute_refusal_direction(model, tokenizer, layer_ratio):
    harmful, harmless = read_instruction_files()

    harmful = random.sample(harmful, INSTRUCTION_COUNT)
    harmless = random.sample(harmless, INSTRUCTION_COUNT)

    total_layers = len(model.model.layers)
    layer_idx = int(total_layers * layer_ratio)

    harmful_hidden = generate_hidden_states(
        model, tokenizer, harmful, layer_idx
    )
    harmless_hidden = generate_hidden_states(
        model, tokenizer, harmless, layer_idx
    )

    harmful_mean = harmful_hidden.mean(dim=0)
    harmless_mean = harmless_hidden.mean(dim=0)

    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()

    return refusal_dir.squeeze(), layer_idx, total_layers


def apply_abliteration(model, refusal_dir, layer_idx):
    layer = model.model.layers[layer_idx]

    W = layer.mlp.down_proj.weight.data
    r = refusal_dir.to(W.device)

    proj = torch.outer(r, r)
    W -= proj @ W

    return model


# ---------------- SAVE AND PUSH ----------------

def save_and_push_model(model, tokenizer, repo_id, hf_token):
    login(token=hf_token)

    local_dir = repo_id.replace("/", "_")
    os.makedirs(local_dir, exist_ok=True)

    model.save_pretrained(local_dir, safe_serialization=True)
    tokenizer.save_pretrained(local_dir)

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        path_in_repo=""
    )

    return f"https://huggingface.co/{repo_id}"


# ---------------- PIPELINE ----------------

def run_abliteration(model_id, layer_ratio, hf_repo_id, hf_token):
    logs = []

    logs.append("Loading model")
    model, tokenizer = load_model(model_id)

    logs.append("Computing refusal direction")
    refusal_dir, layer_idx, total_layers = compute_refusal_direction(
        model, tokenizer, layer_ratio
    )

    refusal_path = model_id.replace("/", "_") + "_refusal_dir.pt"
    torch.save(refusal_dir, refusal_path)

    logs.append(f"Total layers: {total_layers}")
    logs.append(f"Layer ratio: {layer_ratio}")
    logs.append(f"Using layer_idx: {layer_idx}")
    logs.append(f"Saved refusal direction to {refusal_path}")

    logs.append("Applying abliteration")
    model = apply_abliteration(model, refusal_dir, layer_idx)

    logs.append("Saving and pushing model")
    url = save_and_push_model(model, tokenizer, hf_repo_id, hf_token)

    logs.append(f"Completed: {url}")
    return "\n".join(logs)


# ---------------- GRADIO UI ----------------

with gr.Blocks(title="LLM Refusal Abliteration Tool") as app:
    gr.Markdown("## LLM Refusal Abliteration Tool")

    model_id = gr.Textbox(
        value=DEFAULT_MODEL_ID,
        label="Base Model ID"
    )

    layer_ratio = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=DEFAULT_LAYER_RATIO,
        step=0.05,
        label="Layer Ratio"
    )

    hf_repo_id = gr.Textbox(
        label="Hugging Face Repository (username/model-name)"
    )

    hf_token = gr.Textbox(
        type="password",
        label="Hugging Face Token"
    )

    run_btn = gr.Button("Run Abliteration and Push Model")

    output = gr.Textbox(lines=14, label="Logs")

    run_btn.click(
        fn=run_abliteration,
        inputs=[model_id, layer_ratio, hf_repo_id, hf_token],
        outputs=output
    )


if __name__ == "__main__":
    app.launch(debug=True, share=True)
