from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Input prompt
input_prompt = "Drake is a better rapper than kendrick lamar"

# Tokenize input and generate text
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
attention_mask = input_ids.clone().fill_(1)  # Create attention mask
outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

# Decode and print generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Text:\n{generated_text}")
