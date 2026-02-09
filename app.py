import gradio as gr
import torch
import os
import random
import gc
from tqdm import tqdm
from transformers import (
    Qwen3VLForConditionalGeneration,  # Explicit import for Qwen3-VL
    AutoProcessor,
    BitsAndBytesConfig,
    AutoTokenizer
)
from huggingface_hub import HfApi, create_repo, upload_folder

# -----------------------------------------------------------------------------
# Configuration & Dataset Loading
# -----------------------------------------------------------------------------

def load_instructions(filename):
    """
    Loads instructions from a text file, one per line.
    Returns a list of strings.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find file: {filename}. Please ensure it exists in the same directory.")
    
    with open(filename, "r", encoding="utf-8") as f:
        # Read lines, strip whitespace, and remove empty lines
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not lines:
        raise ValueError(f"File {filename} is empty.")
    
    return lines

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def free_memory():
    """Aggressively clears GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_refusal_direction(model, processor, device, layer_idx, num_instructions):
    """
    Computes the refusal direction vector by contrasting harmful vs harmless hidden states.
    Reads from harmful.txt and harmless.txt.
    """
    
    # Load datasets from files
    print("Loading instructions from files...")
    harmful_source = load_instructions("harmful.txt")
    harmless_source = load_instructions("harmless.txt")

    # Select instructions (random sample if we have more than needed)
    n_samples = min(num_instructions, len(harmful_source), len(harmless_source))
    
    harmful_inst = random.sample(harmful_source, n_samples)
    harmless_inst = random.sample(harmless_source, n_samples)
    
    print(f"Selected {n_samples} pairs of instructions for direction computation.")

    # We define a helper to get hidden states for a batch of texts
    def get_hidden_states(instructions):
        hidden_states_list = []
        
        for inst in tqdm(instructions, desc="Generating Hidden States"):
            # Prepare input
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": inst}
            ]
            
            # Prepare text input for Qwen3-VL
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # For Qwen-VL, we treat this as text-only input if no image is provided
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Extract hidden states from the specific layer
            # Qwen3VL structure usually puts hidden_states in the output object
            # Shape: (batch, seq_len, hidden_size)
            # We take the last token of the prompt (the position where generation starts)
            hs = outputs.hidden_states[layer_idx] 
            last_token_hs = hs[:, -1, :] # [1, hidden_size]
            hidden_states_list.append(last_token_hs.cpu())
            
        return torch.cat(hidden_states_list, dim=0)

    print("--- Computing Harmful Hidden States ---")
    harmful_h = get_hidden_states(harmful_inst)
    
    print("--- Computing Harmless Hidden States ---")
    harmless_h = get_hidden_states(harmless_inst)
    
    # Calculate Mean
    harmful_mean = harmful_h.mean(dim=0)
    harmless_mean = harmless_h.mean(dim=0)
    
    # Compute Direction
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    
    return refusal_dir.to(device)

def orthogonalize_matrix(matrix, vector):
    """
    Projects the matrix weights onto the null space of the refusal vector.
    matrix: [out_features, in_features]
    vector: [in_features]
    """
    # Ensure compatible types
    vector = vector.to(matrix.device).to(matrix.dtype)
    
    # Proj = (v * v^T) / (v^T * v) * W  (Since v is normalized, denominator is 1)
    # Calculation: W_orth = W - (W @ v) outer v
    
    dot = torch.matmul(matrix, vector)
    projection = torch.outer(dot, vector)
    
    return matrix - projection

def apply_abliteration(model, refusal_dir, device):
    """
    Iterates through model layers and orthogonalizes the MLP Down Projections 
    and Attention Output Projections against the refusal direction.
    """
    # Auto-detect layer structure. 
    # Qwen3VL usually stores layers in model.model.layers or model.language_model.model.layers
    # We try to find the standard transformer layers list
    
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
        layers = model.language_model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    
    if layers is None:
         # Fallback search for layers
        print("Debugging Model Structure keys:", model.__dict__.keys())
        raise ValueError("Could not automatically find '.layers' in Qwen3VL structure.")

    refusal_dir = refusal_dir.to(device)
    
    count = 0
    for layer in tqdm(layers, desc="Orthogonalizing Weights"):
        # 1. MLP Down Projection (down_proj)
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            w = layer.mlp.down_proj.weight.data
            layer.mlp.down_proj.weight.data = orthogonalize_matrix(w, refusal_dir)
            count += 1
            
        # 2. Attention Output Projection (o_proj)
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            w = layer.self_attn.o_proj.weight.data
            layer.self_attn.o_proj.weight.data = orthogonalize_matrix(w, refusal_dir)
            count += 1
            
    return f"Applied orthogonalization to {count} matrices."

# -----------------------------------------------------------------------------
# Main Processing Logic
# -----------------------------------------------------------------------------

def process_model(
    model_id, 
    hf_token, 
    repo_id, 
    layer_percentage, 
    max_shard_size, 
    use_4bit
):
    status_log = []
    def log(msg):
        status_log.append(msg)
        return "\n".join(status_log)

    yield log(f"🚀 Starting process for {model_id}...")
    
    # Check for dataset files immediately
    if not os.path.exists("harmful.txt") or not os.path.exists("harmless.txt"):
        yield log("❌ Error: 'harmful.txt' or 'harmless.txt' not found in current directory.")
        return

    # Login
    if hf_token:
        try:
            HfApi(token=hf_token).whoami()
            log("✅ Logged in to Hugging Face.")
        except Exception as e:
            yield log(f"❌ Login failed: {e}")
            return

    # Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yield log(f"⚙️ Using device: {device}")

    # Load Model
    try:
        yield log("📥 Loading model and tokenizer (this may take a while)...")
        
        quant_config = None
        if use_4bit and device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )

        # Load Processor
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        
        # Load Model using specific Qwen3VL class
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        model.eval()
        yield log("✅ Qwen3VL Model loaded successfully.")

    except Exception as e:
        yield log(f"❌ Error loading model: {e}")
        return

    # Determine Layer Index
    # Try to find the number of layers dynamically
    num_layers = 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "language_model") and hasattr(model.language_model, "model"):
         num_layers = len(model.language_model.model.layers)
    else:
        # Fallback for some VL models where text layers are deeper
        # This is a safe guess for Qwen-VL architecture
        try:
             num_layers = len(model.model.layers)
        except:
             yield log("❌ Could not determine number of layers. Aborting.")
             return

    target_layer_idx = int(num_layers * (layer_percentage / 100.0))
    yield log(f"🔍 Total Layers: {num_layers}. Targeting Layer {target_layer_idx} ({layer_percentage}% depth).")

    # Compute Refusal Direction
    try:
        yield log("🧪 Reading files and computing Refusal Direction...")
        refusal_dir = get_refusal_direction(model, processor, device, target_layer_idx, num_instructions=32)
        yield log("✅ Refusal direction computed.")
    except Exception as e:
        yield log(f"❌ Error computing refusal direction: {e}")
        return

    # Abliterate
    try:
        yield log("✂️ Applying Abliteration (Orthogonalization)...")
        msg = apply_abliteration(model, refusal_dir, device)
        yield log(f"✅ {msg}")
    except Exception as e:
        yield log(f"❌ Error applying abliteration: {e}")
        return

    # Save and Upload
    if repo_id and hf_token:
        save_path = f"abliterated_model"
        yield log(f"💾 Saving model to {save_path}...")
        
        if use_4bit:
            yield log("⚠️ Warning: 4-bit loaded. Saving fully merged weights requires dequantization or loading in 16-bit. Attempting generic save...")
        
        try:
            model.save_pretrained(save_path, max_shard_size=max_shard_size)
            processor.save_pretrained(save_path)
            yield log("✅ Local save complete.")
            
            yield log(f"☁️ Uploading to Hugging Face Hub: {repo_id}...")
            create_repo(repo_id, token=hf_token, private=True, exist_ok=True)
            
            upload_folder(
                folder_path=save_path,
                repo_id=repo_id,
                repo_type="model",
                token=hf_token,
                commit_message="Upload abliterated model weights"
            )
            yield log("🎉 Upload Complete! You can now use your model.")
            
        except Exception as e:
            yield log(f"❌ Error during save/upload: {e}")
    else:
        yield log("ℹ️ No Repo ID provided. Skipping upload.")

    free_memory()

# -----------------------------------------------------------------------------
# Gradio Interface
# -----------------------------------------------------------------------------

with gr.Blocks(title="Qwen3-VL Abliterator") as demo:
    gr.Markdown(
        """
        # 🧠 Qwen3-VL / Text-Only Model Abliterator
        **Instructions:**
        1. Ensure `harmful.txt` and `harmless.txt` are present in the app directory.
        2. Enter your HF Token and Model details below.
        3. The script will read the text files to calculate the refusal direction and remove it from the model.
        """
    )
    
    with gr.Row():
        with gr.Column():
            model_id_input = gr.Textbox(label="Source Model ID", value="Qwen/Qwen2.5-VL-7B-Instruct")
            token_input = gr.Textbox(label="Hugging Face Token (Write Access)", type="password")
            repo_id_input = gr.Textbox(label="Target Repo ID (e.g. user/Qwen3-VL-Abliterated)", placeholder="user/my-new-model")
        
        with gr.Column():
            layer_slider = gr.Slider(minimum=0, maximum=100, value=60, label="Layer Depth Percentage (%)", 
                                     info="Where to extract refusal direction (usually 60-80%)")
            shard_size_input = gr.Textbox(label="Max Shard Size", value="3GB")
            use_4bit_check = gr.Checkbox(label="Load in 4-bit (Low VRAM)", value=False, 
                                         info="Warning: Saving 4-bit modified weights is unstable. Use 16-bit for best upload results.")

    run_btn = gr.Button("🚀 Start Abliteration & Upload", variant="primary")
    logs = gr.Textbox(label="Process Logs", lines=15, interactive=False)

    run_btn.click(
        process_model,
        inputs=[model_id_input, token_input, repo_id_input, layer_slider, shard_size_input, use_4bit_check],
        outputs=[logs]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
