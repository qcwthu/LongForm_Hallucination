from vllm import LLM, SamplingParams
import json
import argparse
import os
import random
import numpy as np
import torch
from typing import List

tensor_parallel=8

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load(file_path: str) -> List[str]:
    prompts = []
    with open(file_path, 'r') as file:
        for line in file:
            prompts.append(json.loads(line)['prompt'])
    return prompts

set_seed(42)
folderpath_concept = "../Knowledge-Boundary/longfact/longfact-concepts_gpt4_01-10-2024_noduplicates"
allprompts = []
for file in os.listdir(folderpath_concept):
    allprompts.extend(load(os.path.join(folderpath_concept, file)))

folderpath_object = "../Knowledge-Boundary/longfact/longfact-objects_gpt4_01-12-2024_noduplicates"
for file in os.listdir(folderpath_object):
    allprompts.extend(load(os.path.join(folderpath_object, file)))

print("size before sampling: ", len(allprompts))
samplingnum = len(allprompts)
allprompts = random.sample(allprompts, samplingnum)
print("size after sampling: ", len(allprompts))
print(allprompts[0])
print(allprompts[-1])
# exit -1

model = "meta-llama/Meta-Llama-3-70B-Instruct"
llm = LLM(model, tensor_parallel_size=tensor_parallel)
tokenizer = llm.get_tokenizer()

processedprompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False
    ) for prompt in allprompts
]

temperature = 1.0
top_p = 0.9
num_responses = 20
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=1024,
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    use_beam_search=False,
    n=num_responses
)

outputs = llm.generate(processedprompts, sampling_params)

resultpath = "./SelfCheckGPT/longfact_selfcheckgpt_stochastic_samples.json"
print(resultpath)
with open(resultpath, 'w') as fw:
    for output in outputs:
        prompt = output.prompt
        all_responses = []
        all_generated_text = output.outputs
        for i in range(len(all_generated_text)):
            one_response = all_generated_text[i].text
            all_responses.append(one_response)
        assert len(all_responses) == num_responses
        onejsonline = {'prompt': prompt, 'response': all_responses}
        line = json.dumps(onejsonline)
        fw.write(line+"\n")

