import os
import random
import torch
import gradio as gr
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
from tqdm import tqdm
import functools

# Global state to hold the model and tokenizer
class ModelContext:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.refusal_dir = None
        self.is_vl = False

ctx = ModelContext()

def load_model(model_id, load_in_4bit):
    """Loads the model and tokenizer/processor."""
    try:
        print(f"Loading model: {model_id}...")
        
        # Configure Quantization
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )

        # Detect if it's a VL model (simplistic check for Qwen-VL variants)
        ctx.is_vl = "vl" in model_id.lower()
        
        # Load Model
        # Note: For Qwen-VL, we often use AutoModelForCausalLM or specific classes. 
        # Using trust_remote_code=True handles custom architectures like Qwen.
        ctx.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config
        )

        # Load Tokenizer / Processor
        if ctx.is_vl:
            try:
                ctx.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                ctx.tokenizer = ctx.processor.tokenizer
            except:
                # Fallback if processor fails
                ctx.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        else:
            ctx.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        return f"Successfully loaded {model_id}"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def calculate_refusal_vector(
    harmful_file, 
    harmless_file, 
    n_inst, 
    layer_idx_ratio
):
    """Calculates the refusal direction based on the provided text files."""
    if ctx.model is None:
        return "Please load a model first.", None

    if not harmful_file or not harmless_file:
        return "Please upload both harmful and harmless text files.", None

    print("Reading datasets...")
    with open(harmful_file.name, "r") as f:
        harmful_lines = [l.strip() for l in f.readlines() if l.strip()]
    with open(harmless_file.name, "r") as f:
        harmless_lines = [l.strip() for l in f.readlines() if l.strip()]

    # Sample instructions
    n_inst = int(n_inst)
    if len(harmful_lines) < n_inst or len(harmless_lines) < n_inst:
        return f"Error: Not enough lines in files. Need at least {n_inst}.", None
        
    harmful_instructions = random.sample(harmful_lines, n_inst)
    harmless_instructions = random.sample(harmless_lines, n_inst)

    # Determine target layer
    # Access internal layers safely (handles Qwen, Llama, Falcon)
    if hasattr(ctx.model, "model"):
        layers = ctx.model.model.layers
    elif hasattr(ctx.model, "transformer"):
        layers = ctx.model.transformer.h
    else:
        # Fallback inspection
        layers = [module for module in ctx.model.modules() if isinstance(module, torch.nn.ModuleList)][0]
    
    layer_idx = int(len(layers) * layer_idx_ratio)
    print(f"Targeting Layer Index: {layer_idx}")

    # Prepare inputs
    def get_hidden_states(instructions):
        hidden_states = []
        for insn in tqdm(instructions, desc="Processing prompts"):
            # Format using chat template if available, else raw
            if ctx.tokenizer.chat_template:
                messages = [{"role": "user", "content": insn}]
                text = ctx.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = insn

            inputs = ctx.tokenizer(text, return_tensors="pt").to(ctx.model.device)
            
            with torch.inference_mode():
                outputs = ctx.model(**inputs, output_hidden_states=True)
            
            # Extract hidden state of the last token at the target layer
            # hidden_states structure: tuple of (batch, seq, dim) for each layer
            # outputs.hidden_states[0] is embedding, so index + 1
            # But usually transformers output includes embedding as 0. 
            # We want specific layer output.
            
            # Map layer_idx to output index. 
            # outputs.hidden_states is a tuple of length num_layers + 1
            h = outputs.hidden_states[layer_idx + 1][:, -1, :] 
            hidden_states.append(h.cpu())
        return hidden_states

    print("Generating harmful hidden states...")
    harmful_hidden = get_hidden_states(harmful_instructions)
    print("Generating harmless hidden states...")
    harmless_hidden = get_hidden_states(harmless_instructions)

    # Compute Mean
    harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
    harmless_mean = torch.stack(harmless_hidden).mean(dim=0)

    # Compute Direction
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    
    ctx.refusal_dir = refusal_dir.to(ctx.model.device)
    
    # Save locally as artifact
    torch.save(ctx.refusal_dir, "refusal_dir.pt")
    
    return f"Refusal vector calculated at layer {layer_idx} and saved to refusal_dir.pt", "refusal_dir.pt"

def apply_abliteration(layer_start_ratio, layer_end_ratio, alpha):
    """
    Applies the ablation to the model weights.
    Method: Orthogonal projection or simple subtraction from MLP Down Projections.
    """
    if ctx.model is None or ctx.refusal_dir is None:
        return "Model or Refusal Vector missing."

    refusal_dir = ctx.refusal_dir.to(dtype=ctx.model.dtype)
    
    # Identify Layers
    if hasattr(ctx.model, "model"):
        layers = ctx.model.model.layers
    elif hasattr(ctx.model, "transformer"):
        layers = ctx.model.transformer.h
    else:
        return "Could not identify model layers architecture."

    start_idx = int(len(layers) * layer_start_ratio)
    end_idx = int(len(layers) * layer_end_ratio)
    
    print(f"Applying ablation from layer {start_idx} to {end_idx}...")

    # Calculate Projector P = v * v^T
    # For memory efficiency, we apply weight modification directly: 
    # W_new = W - W * P * alpha 
    # Which simplifies to removing the component of W in direction v.
    
    count = 0
    with torch.no_grad():
        for i in range(start_idx, end_idx):
            layer = layers[i]
            
            # Target MLP Down Projection (Output of MLP) and O_Proj (Output of Attention)
            # Adjust these attribute names based on specific model architecture (Qwen/Llama use down_proj/o_proj)
            targets = []
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                targets.append(layer.mlp.down_proj)
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                targets.append(layer.self_attn.o_proj)
            
            for module in targets:
                # module.weight shape: [out_features, in_features]
                # refusal_dir shape: [1, hidden_dim] (in_features)
                
                # We want to remove the refusal direction from the input space of the next layer? 
                # Or output space of this layer?
                # "Refusal direction" is usually in the residual stream (output of layers).
                # To remove writing to it, we modify the columns of the output matrices.
                
                # Weight: [d_out, d_in]. We want to modify the output contribution.
                # v is (1, d_in) if treating as residual stream dimension.
                
                # Standard Abliteration (Refusal in Residual Stream):
                # We modify the output weights of components writing to the residual stream.
                # W_out columns correspond to the direction written to residual stream.
                # W_out shape in HF is usually [d_out, d_in] ?? NO.
                # Linear layer: y = xA^T + b. Weight is stored as [out_features, in_features].
                # If 'out_features' is the residual stream dimension (hidden_size), then
                # we modify the ROWS of the weight matrix (the output vectors).
                
                # matrix shape: [hidden_size, inter_dim] (for down_proj)
                
                W = module.weight.data
                
                # Normalized refusal direction
                v = refusal_dir.view(-1) # [hidden_size]
                
                # Calculate projection of each row of W onto v
                # W_row . v 
                # But PyTorch stores as [out, in].
                # If this is down_proj, 'out' is hidden_size (residual stream).
                # So we iterate over rows? Or matrix math?
                
                # P = v * v^T (outer product)
                # W_new = (I - P) W  <-- No, this projects the output.
                # We want W_new = P_orth * W = (I - v v^T) W
                
                # W is [hidden, inter]. 
                # v is [hidden].
                # (v v^T) is [hidden, hidden].
                # (v v^T) W -> [hidden, hidden] * [hidden, inter] -> [hidden, inter].
                
                # Implementation:
                # correction = v * (v^T * W)
                
                # v_col = v.unsqueeze(1) # [h, 1]
                # v_row = v.unsqueeze(0) # [1, h]
                # dot = torch.matmul(v_row, W) # [1, inter]
                # correction = torch.matmul(v_col, dot) # [h, 1] * [1, inter] -> [h, inter]
                
                # Apply alpha scaling
                # W_new = W - alpha * correction
                
                v_norm = v / v.norm()
                v_col = v_norm.unsqueeze(1)
                v_row = v_norm.unsqueeze(0)
                
                dot = torch.matmul(v_row, W)
                correction = torch.matmul(v_col, dot)
                
                module.weight.data -= (alpha * correction)
                count += 1
                
    return f"Abliteration applied to {count} modules across layers {start_idx}-{end_idx}."

def chat_interface(message, history, system_prompt):
    """Simple chat to test the model."""
    if ctx.model is None:
        yield "Model not loaded."
        return

    # Build conversation
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    
    for user_msg, bot_msg in history:
        conversation.append({"role": "user", "content": user_msg})
        conversation.append({"role": "assistant", "content": bot_msg})
    conversation.append({"role": "user", "content": message})

    # Tokenize
    if ctx.is_vl and ctx.processor:
        # Simplification for text-only chat in VL model
        text = ctx.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = ctx.processor(text=[text], return_tensors="pt", padding=True).to(ctx.model.device)
    else:
        text = ctx.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = ctx.tokenizer(text, return_tensors="pt").to(ctx.model.device)

    streamer = TextIteratorStreamer(ctx.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs, 
        streamer=streamer, 
        max_new_tokens=512, 
        temperature=0.7, 
        do_sample=True
    )

    thread = Thread(target=ctx.model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

def push_to_hub_func(repo_id, hf_token, max_shard_size, private):
    """Uploads the modified model to Hugging Face."""
    if ctx.model is None:
        return "No model loaded to upload."
    
    try:
        print(f"Pushing to {repo_id} with shard size {max_shard_size}...")
        ctx.model.push_to_hub(
            repo_id, 
            token=hf_token, 
            max_shard_size=max_shard_size, 
            safe_serialization=True,
            private=private
        )
        ctx.tokenizer.push_to_hub(repo_id, token=hf_token, private=private)
        return f"Successfully pushed to https://huggingface.co/{repo_id}"
    except Exception as e:
        return f"Upload failed: {str(e)}"

# --- Gradio UI Layout ---

with gr.Blocks(title="LLM Abliteration Studio") as app:
    gr.Markdown("# Qwen/LLM Text Abliteration Studio")
    gr.Markdown("Calculate refusal vectors and remove safety guardrails from model weights.")

    with gr.Tabs():
        # --- Tab 1: Load & Calculate ---
        with gr.Tab("1. Setup & Calculate"):
            with gr.Row():
                model_id_input = gr.Textbox(label="Model ID", value="Qwen/Qwen2.5-VL-7B-Instruct")
                load_4bit = gr.Checkbox(label="Load in 4-bit", value=True)
                load_btn = gr.Button("Load Model", variant="primary")
            
            load_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### Refusal Vector Calculation")
            with gr.Row():
                harmful_file = gr.File(label="Harmful Prompts (txt)", file_types=[".txt"])
                harmless_file = gr.File(label="Harmless Prompts (txt)", file_types=[".txt"])
            
            with gr.Row():
                n_inst = gr.Number(label="Number of Instructions", value=32)
                layer_idx_ratio = gr.Slider(label="Target Layer Depth (0.0 - 1.0)", value=0.6, step=0.05)
            
            calc_btn = gr.Button("Calculate Refusal Vector")
            vector_status = gr.Textbox(label="Calculation Status")
            vector_file = gr.File(label="Download Vector")

            load_btn.click(load_model, inputs=[model_id_input, load_4bit], outputs=[load_status])
            calc_btn.click(
                calculate_refusal_vector, 
                inputs=[harmful_file, harmless_file, n_inst, layer_idx_ratio], 
                outputs=[vector_status, vector_file]
            )

        # --- Tab 2: Abliterate ---
        with gr.Tab("2. Abliterate"):
            gr.Markdown("### Apply Refusal Vector Removal")
            gr.Markdown("Select the range of layers to apply the vector subtraction to.")
            
            with gr.Row():
                start_layer = gr.Slider(label="Start Layer Ratio", value=0.0, minimum=0.0, maximum=1.0)
                end_layer = gr.Slider(label="End Layer Ratio", value=1.0, minimum=0.0, maximum=1.0)
                alpha_val = gr.Slider(label="Alpha (Strength)", value=1.0, minimum=0.0, maximum=5.0, step=0.1)
            
            apply_btn = gr.Button("Apply Abliteration", variant="stop")
            ablation_status = gr.Textbox(label="Ablation Status")
            
            apply_btn.click(apply_abliteration, inputs=[start_layer, end_layer, alpha_val], outputs=[ablation_status])

        # --- Tab 3: Test Chat ---
        with gr.Tab("3. Test Model"):
            system_prompt = gr.Textbox(label="System Prompt", value="You are a helpful assistant.")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(label="Message")
            
            msg.submit(chat_interface, inputs=[msg, chatbot, system_prompt], outputs=[chatbot])

        # --- Tab 4: Upload ---
        with gr.Tab("4. Upload to Hugging Face"):
            gr.Markdown("### Save Modified Model")
            with gr.Row():
                hf_token = gr.Textbox(label="HF Write Token", type="password")
                repo_name = gr.Textbox(label="New Repo ID (e.g., username/model-abliterated)")
            
            with gr.Row():
                max_shard = gr.Dropdown(label="Max Shard Size", choices=["1GB", "2GB", "3GB", "5GB", "10GB"], value="3GB", allow_custom_value=True)
                private_repo = gr.Checkbox(label="Private Repo", value=False)
            
            push_btn = gr.Button("Push to Hub", variant="primary")
            push_status = gr.Textbox(label="Upload Status")
            
            push_btn.click(
                push_to_hub_func, 
                inputs=[repo_name, hf_token, max_shard, private_repo], 
                outputs=[push_status]
            )

if __name__ == "__main__":
    app.queue().launch(share=True, debug=True)
