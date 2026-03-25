"""
Test if Qwen base model already knows finance-alpaca content.
This checks if we're actually teaching the model new information.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading Qwen 2.5-1.5B-Instruct base model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Test question from finance-alpaca dataset
finance_question = "For a car, what scams can be plotted with 0% financing vs rebate?"

print("\n" + "="*80)
print("TESTING BASE MODEL (before finetuning)")
print("="*80)
print(f"\nQuestion: {finance_question}\n")

# Format like our training data
prompt = f"##Human:{finance_question}\n\n##Response:"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

# Generate response
with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

# Decode and show
output = tokenizer.decode(generated[0], skip_special_tokens=True)
print("BASE MODEL RESPONSE:")
print("-" * 80)
print(output)
print("-" * 80)

# Also test with native Qwen format (no ##Human/##Response markers)
print("\n" + "="*80)
print("TESTING WITH NATIVE QWEN FORMAT")
print("="*80)
print(f"\nQuestion: {finance_question}\n")

input_ids2 = tokenizer.encode(finance_question, return_tensors='pt').to(model.device)

with torch.no_grad():
    generated2 = model.generate(
        input_ids2,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

output2 = tokenizer.decode(generated2[0], skip_special_tokens=True)
print("NATIVE FORMAT RESPONSE:")
print("-" * 80)
print(output2)
print("-" * 80)

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("If the base model gives generic/vague answers, finance-alpaca will teach new info.")
print("If it gives detailed scam explanations, the dataset won't add much value.")
print("="*80)
