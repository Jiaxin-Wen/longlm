from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

load_path = '/dataset/f1d6ea5b/wenjiaxin/lot/results/lot_large_esc_deepspeed/checkpoint-1000'
device = 'cuda:0'

tokenizer = T5Tokenizer.from_pretrained(load_path)
tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>"%k for k in range(100)]})

text = '发生什么事了<sep><extra_id_1>'
input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids

print('input_ids = ', input_ids)

print('tokens = ', tokenizer.decode(input_ids.item()))

enc_input_ids = input_ids.to(device)

dec_input = '<s>'
dec_input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dec_input))], dtype=torch.long).to(device)
print('dec_input_ids = ', dec_input_ids)

# load model
model = T5ForConditionalGeneration.from_pretrained(load_path).to(device)
print(f"loaded model from : {load_path}")
gen = model.generate(enc_input_ids, decoder_input_ids=dec_input_ids, do_sample=True, max_length=128, top_p=0.9, temperature=0.7, decoder_start_token_id=1)

print('gen = ', gen)
print('gen = ', tokenizer.decode(gen))