"""
Test script for Gemma-3-1B-IT model
"""
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

# Model path
model_path = "/share/disk/llm_cache/gemma-3-1b-it"

print(f"Loading Gemma model from {model_path}...")

try:
    # Try AutoTokenizer first (for text-only models)
    # If this fails, we'll try AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    processor = tokenizer  # Use tokenizer as processor
    print("✅ Tokenizer loaded successfully")
    
    # Load model on GPU 4
    # Use Gemma3ForCausalLM for gemma-3-1b-it (text-only model)
    # This matches the official Hugging Face documentation
    device_index = 4
    model = Gemma3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device_index},
        trust_remote_code=True
    ).eval()
    print(f"✅ Model loaded successfully on cuda:{device_index}")
    
    # Prepare messages for chat template
    # For gemma-3-1b-it, use the format from official documentation
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello! Please introduce yourself."}]
            }
        ]
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    
    # Move to device and convert to bfloat16 (matching official example)
    inputs = inputs.to(model.device).to(torch.bfloat16)
    
    print(f"✅ Input processed, input_ids shape: {inputs['input_ids'].shape}")
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate response
    print("Generating response...")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode the generated tokens (skip the input part)
    generated_tokens = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print("\n" + "="*50)
    print("Generated Response:")
    print("="*50)
    print(generated_text)
    print("="*50)
    print("\n✅ Test completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()

