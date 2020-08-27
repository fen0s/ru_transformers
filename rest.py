import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from run_generation import sample_sequence
from yt_encoder import YTEncoder
from transformers import GPT2LMHeadModel
import threading
import regex as re

from os import environ
device = environ.get('DEVICE', 'cuda:0')

flavor_id = device + environ.get('INSTANCE', ':0')
from tendo import singleton
me = singleton.SingleInstance(flavor_id=flavor_id)

model_path = "../gpt3/checkpoint-" + input("checkpoint number after -")

tokenizer = YTEncoder.from_pretrained(model_path)

model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)
model.eval()

from apex import amp
[model] = amp.initialize([model], opt_level='O2')

def get_sample(model, prompt, length:int, num_samples:int, allow_linebreak:bool):
    print(prompt)
   
    filter_n = tokenizer.encode('\n')[-1:]
    filter_single = [1] + tokenizer.encode('[')[-1:] + tokenizer.encode('(')[-1:]
    filter_single += [] if allow_linebreak else filter_n

    context_tokens = tokenizer.encode(prompt)
    out = sample_sequence(
        model=model,
        context=context_tokens,
        length=length,
        temperature=0.8,
        top_k=0,
        top_p=0.9,
        device=device,
        filter_single=filter_single,
        filter_double=filter_n,
        num_samples=num_samples,
    ).to('cpu')

    prompt = tokenizer.decode(context_tokens)
    len_prompt = len(prompt)
   
    replies = [out[item, :].tolist() for item in range(len(out))]
    text = [tokenizer.decode(item)[len_prompt:] for item in replies]
    reg_text = [re.match(r'[\w\W]*[\.!?]\n', item) for item in text]
    reg_text2 = [re.match(r'[\w\W]*[\.!?]', item) for item in text]
    result = [reg_item[0] if reg_item else reg_item2[0] if reg_item2 else item for reg_item, reg_item2, item in zip(reg_text, reg_text2, text)]
    print(result[0])
    return result
promppt = input("what to prompt with")
get_sample(model, promppt, 500, 1, True)