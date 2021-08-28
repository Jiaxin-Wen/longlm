import sys
import os
import json
import argparse
from unicodedata import category

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_path", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)

    parser.add_argument("--enc_seq_length", type=int, default=128)
    parser.add_argument("--dec_seq_length", type=int, default=128)
    
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--device", type=int)

    args = parser.parse_args()

    args.device = f"cuda:{args.device}"
    print('device = ', args.device)
    return args


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    chrs = (chr(i) for i in range(sys.maxunicode + 1))
    punctuation = set(c for c in chrs if category(c).startswith("P"))
    for uchar in ustring.replace("...", "…"):
        inside_code=ord(uchar)
        if uchar in punctuation:
            if inside_code == 32:
                inside_code = 12288
            elif inside_code >= 32 and inside_code <= 126:
                inside_code += 65248
        rstring += chr(inside_code)
    return rstring

def process(token_list, tokenizer):
    string = tokenizer.convert_ids_to_tokens(token_list, skip_special_tokens=False)
    string = "".join(string)
    string = string[:string.find("</s>")].replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
    string = string.replace('▁', '')
    # print(string)
    for i in range(100):
        string = string.replace("<extra_id_%d>"%i, "")
    # string = "".join(string.strip().split())
    # string = strB2Q(string)
    return string


def generate(args):
    # load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.load_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<extra_id_%d>"%k for k in range(100)]})

    # load model
    model = T5ForConditionalGeneration.from_pretrained(args.load_path).to(args.device)
    print(f"loaded model from : {args.load_path}")

    # load data
    with open(args.input_path + '.source', 'r', encoding='utf-8') as f:
        input = [line.strip() for line in f.readlines()]
    print(f"loaded input data from : {args.input_path + '.source'}")
    with open(args.input_path + '.target', 'r', encoding='utf-8') as f:
        target = [line.strip() for line in f.readlines()]
    print(f"loaded input data from : {args.input_pat + '.target'}")


    print(f'start generate to : {args.output_path}')
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    start, end = 0, 0
    generate = []
    with open(args.output_path + 'generate.txt', 'w', encoding='utf-8') as f:
        with torch.no_grad():
            while end < len(input):
                start, end = end, min(end + args.batch_size, len(input))
                enc_input_ids = tokenizer(input[start: end], return_tensors="pt", padding=True, truncation=True, max_length=args.enc_seq_length).input_ids.to(args.device)
                dec_input = '<s>'
                dec_input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dec_input)) for _ in range(end-start)], dtype=torch.long).to(args.device)
                gen = model.generate(enc_input_ids, decoder_input_ids=dec_input_ids, do_sample=True, max_length=args.dec_seq_length, top_p=0.9, temperature=0.7, decoder_start_token_id=1)
                gen = [process(i, tokenizer) for i in gen]
                generate += gen
                print(f'start = {start}, end = {end}')

                for output in gen:
                    f.write(output)
                    f.write('\n')
                f.flush()


    # TODO: cal metric
    with open(args.output_path + 'metric.json', 'w', encoding='utf-8') as f:
        pass
    

if __name__ == "__main__":
    args = get_args()
    generate(args)