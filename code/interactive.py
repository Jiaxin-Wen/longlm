import sys
import os
import json
import argparse
from unicodedata import category

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

from myMetrics import Metric


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_path", type=str, default="/dataset/f1d6ea5b/wenjiaxin/lot/results/lot_large_esc_manual0831")

    parser.add_argument("--enc_seq_length", type=int, default=128)
    parser.add_argument("--dec_seq_length", type=int, default=128)

    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    args.device = f"cuda:{args.device}"
    print('device = ', args.device)
    return args

def process(token_list, tokenizer):
    string = tokenizer.convert_ids_to_tokens(token_list, skip_special_tokens=False)
    string = "".join(string)
    string = string[:string.find("</s>")].replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
    string = string.replace('▁', '')
    for i in range(100):
        string = string.replace("<extra_id_%d>"%i, "")
    return string


def interactive(args):
    # load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.load_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>"%k for k in range(100)]})

    # load model
    model = T5ForConditionalGeneration.from_pretrained(args.load_path).to(args.device)
    print(f"loaded model from : {args.load_path}")

    context = []
    with torch.no_grad():
        while True:
            user_input = input("user: ")
            if user_input == 'restart':
                context = []
                continue
            context.append(user_input)
            enc_input_tokens = '<sep>'.join(context) + '<extra_id_1>'
            # start, end = end, min(end + args.batch_size, len(input))
            enc_input_ids = tokenizer(enc_input_tokens, return_tensors="pt", padding=True, truncation=True, max_length=args.enc_seq_length).input_ids.to(args.device)
            dec_input = '<s>'
            dec_input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dec_input))], dtype=torch.long).to(args.device)
            gen = model.generate(enc_input_ids, decoder_input_ids=dec_input_ids, do_sample=True, max_length=args.dec_seq_length, top_p=0.9, temperature=0.7, decoder_start_token_id=1)
            gen = [process(i, tokenizer) for i in gen]
            context += gen
            print("system: ", gen[0])


if __name__ == "__main__":
    args = get_args()
    interactive(args)