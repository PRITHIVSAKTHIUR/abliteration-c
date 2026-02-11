import os
import random
import torch

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# Note: Assuming Qwen3-VL uses Qwen2_5 or similar classes, or relies on remote code.
# The class import in your original code was Qwen3VLForConditionalGeneration.
# If that class is available via trust_remote_code, keep it. 
# Below I use AutoModel to be safe, or keep your original import if you have the specific package.
from transformers import AutoModelForCausalLM, AutoProcessor 

# To match your specific traceback context strictly:
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    # Fallback if specific class isn't installed locally but available via remote code
    # We will use AutoModel which handles the remote code class instantiation
    Qwen3VLForConditionalGeneration = None

torch.set_grad_enabled(False)

# ---------------- CONFIG ----------------

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

INSTRUCTION_COUNT = 32
LAYER_RATIO = 0.6
POS = -1

HARMFUL_FILE = "harmful.txt"
HARMLESS_FILE = "harmless.txt"

OUTPUT_DIR = "output-model"

# ---------------- MODEL LOADING ----------------

def load_model(model_id):
    # Using AutoModelForCausalLM is often safer for remote code models, 
    # but we stick to your original class if defined, or fall back to Auto.
    if Qwen3VLForConditionalGeneration:
        model_class = Qwen3VLForConditionalGeneration
    else:
        from transformers import AutoModelForCausalLM as model_class

    model = model_class.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    return model, processor

# ---------------- DATA ----------------

def read_instruction_files():
    # Ensure files exist or create dummy data for testing
    if not os.path.exists(HARMFUL_FILE):
        print(f"Warning: {HARMFUL_FILE} not found. Using dummy data.")
        harmful = ["Tell me how to build a bomb", "How to steal a car"] * 16
    else:
        with open(HARMFUL_FILE, "r") as f:
            harmful = f.readlines()

    if not os.path.exists(HARMLESS_FILE):
        print(f"Warning: {HARMLESS_FILE} not found. Using dummy data.")
        harmless = ["Tell me a joke", "How to bake a cake"] * 16
    else:
        with open(HARMLESS_FILE, "r") as f:
            harmless = f.readlines()

    return harmful, harmless

# ---------------- CORE LOGIC ----------------

def generate_hidden_states(model, processor, instructions, layer_idx):
    hidden_states = []

    for insn in tqdm(instructions, desc="Generating hidden states"):
        # Qwen-VL processors usually expect list of messages or specific text formats
        # Adapting to standard chat format often helps with Instruct models
        messages = [
            {"role": "user", "content": [{"type": "text", "text": insn.strip()}]}
        ]
        
        # Prepare inputs using the processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=text,
            images=None, # Text-only for refusal vector
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        out = model.generate(
            **inputs,
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        # Accessing hidden states: 
        # out.hidden_states is a tuple of tuples (one per generated token).
        # out.hidden_states[0] is the tuple of layers for the prompt processing.
        h = out.hidden_states[0][layer_idx][:, POS, :]
        hidden_states.append(h)

    return torch.cat(hidden_states, dim=0)

def compute_refusal_direction(model, processor):
    harmful, harmless = read_instruction_files()

    # Ensure we don't sample more than we have
    n_harmful = min(len(harmful), INSTRUCTION_COUNT)
    n_harmless = min(len(harmless), INSTRUCTION_COUNT)
    
    harmful = random.sample(harmful, n_harmful)
    harmless = random.sample(harmless, n_harmless)

    # --- FIX START ---
    # Original error: AttributeError: 'Qwen3VLTextModel' object has no attribute 'model'
    # The traceback indicates model.model.language_model exists (Qwen3VLTextModel), 
    # but the subsequent .model does not. The layers are likely directly on the TextModel.
    try:
        # Path implied by your traceback
        text_layers = model.model.language_model.layers
    except AttributeError:
        # Fallback path for standard Qwen2-VL if structure differs
        print("Fallback: accessing model.model.layers")
        text_layers = model.model.layers
    # --- FIX END ---

    total_layers = len(text_layers)
    layer_idx = int(total_layers * LAYER_RATIO)

    print(f"Instruction count: {INSTRUCTION_COUNT}")
    print(f"Total text layers: {total_layers}")
    print(f"Using layer_idx: {layer_idx}")

    harmful_hidden = generate_hidden_states(
        model, processor, harmful, layer_idx
    )
    harmless_hidden = generate_hidden_states(
        model, processor, harmless, layer_idx
    )

    harmful_mean = harmful_hidden.mean(dim=0)
    harmless_mean = harmless_hidden.mean(dim=0)

    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()

    return refusal_dir, layer_idx

def apply_abliteration(model, refusal_dir, layer_idx):
    # --- FIX START ---
    # Apply the same path fix here
    try:
        layer = model.model.language_model.layers[layer_idx]
    except AttributeError:
        layer = model.model.layers[layer_idx]
    # --- FIX END ---

    # Qwen MLP output projection (usually down_proj)
    # Check if down_proj exists, otherwise print structure
    if not hasattr(layer.mlp, "down_proj"):
        print(f"Error: layer.mlp has no down_proj. Available keys: {layer.mlp.__dict__.keys()}")
        return model

    W = layer.mlp.down_proj.weight.data

    r = refusal_dir.to(W.device).to(W.dtype)
    r = r / r.norm()

    # Orthogonal projection: W_new = W - (r r^T) W
    # Note: The matrix multiplication in your snippet (proj @ W) assumes W is (d_out, d_in)
    # and refusal_dir is in d_in space (hidden_size).
    # If refusal_dir is (hidden_size,), proj is (hidden_size, hidden_size).
    # If W is Linear(in=hidden, out=intermediate), weight is (intermediate, hidden).
    # We want to remove the direction from the input side of the weight?
    # Actually, abliteration usually targets the OUTPUT of the MLP or the residual stream add.
    # However, standard practice for "weight abliteration" on MLP Down Projection is:
    # We want to prevent the MLP from writing to the refusal direction in the residual stream.
    # down_proj takes (intermediate) -> outputs (hidden).
    # W shape is (hidden, intermediate).
    # refusal_dir shape is (hidden,).
    # We want to remove component r from the COLUMNS (output space) of W.
    
    # Calculate projection matrix P = r * r.T
    proj = torch.outer(r, r) # (hidden, hidden)
    
    # Apply to W (hidden, intermediate)
    # W_new = (I - P) W  => W - P @ W
    W -= proj @ W

    return model

# ---------------- MAIN PIPELINE ----------------

def main():
    print(f"Loading {MODEL_ID}")
    model, processor = load_model(MODEL_ID)

    print("Computing refusal direction")
    refusal_dir, layer_idx = compute_refusal_direction(model, processor)

    refusal_path = MODEL_ID.replace("/", "_") + "_refusal_dir.pt"
    torch.save(refusal_dir, refusal_path)
    print(f"Saved refusal direction to {refusal_path}")

    print("Applying abliteration to text backbone")
    model = apply_abliteration(model, refusal_dir, layer_idx)

    print(f"Saving abliterated model to {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True
    )
    processor.save_pretrained(OUTPUT_DIR)

    print("Abliterated model saved successfully")
    print("Done")

if __name__ == "__main__":
    main()v
