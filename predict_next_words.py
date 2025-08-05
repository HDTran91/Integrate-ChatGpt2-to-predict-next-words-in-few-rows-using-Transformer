from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")

#load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#tao input cho model
model_input = tokenizer("Today is a bad day", return_tensors="pt")

# predict 40 next words
output = model.generate(**model_input, max_new_tokens=40, do_sample=True, temperature=0.7)

# print results

print(tokenizer.decode(output[0], skip_special_tokens=True))
