import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union
from transformers import logging
logging.set_verbosity_error()
import spacy
from typing import Dict, List, Set, Tuple, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

class SelfCheckLLMPrompt:
    """
    SelfCheckGPT (LLM Prompt): Checking LLM's text against its own sampled texts via open-source LLM prompting
    """
    def __init__(
        self,
        model: str = None
    ):
        model = model if model is not None else LLMPromptConfig.model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.model.eval()
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 0.0, 'no': 1.0, 'n/a': 0.5}
        self.not_defined_text = set()
        print(f"SelfCheck-LLMPrompt ({model}) initialized to device-auto")

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
        verbose: bool = False,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param verson: bool -- if True tqdm progress bar will be shown
        :return sent_scores: sentence-level scores
        """
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        scores = np.zeros((num_sentences, num_samples))
        disable = not verbose
        for sent_i in tqdm(range(num_sentences), disable=disable):
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 

                prompt = self.prompt_template.format(context=sample, sentence=sentence)
                # print(prompt)
                input_tokens_chat = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False
                )
                # print(input_tokens_chat)
                
                inputs = self.tokenizer(input_tokens_chat, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
                generate_ids = self.model.generate(
                    inputs,
                    max_new_tokens=5,
                    eos_token_id=self.terminators,
                    do_sample=False
                )
                # print(generate_ids)
                newoutput = generate_ids[0][inputs.shape[-1]:]
                # print(newoutput)
                output_text = self.tokenizer.decode(
                    newoutput, skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                # print(output_text)
                score_ = self.text_postprocessing(output_text)
                scores[sent_i, sample_i] = score_
        scores_per_sentence = scores.mean(axis=-1)
        return scores_per_sentence

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        Yes -> 0.0
        No  -> 1.0
        everything else -> 0.5
        """
        # tested on Llama-2-chat (7B, 13B) --- this code has 100% coverage on wikibio gpt3 generated data
        # however it may not work with other datasets, or LLMs
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            if text not in self.not_defined_text:
                print(f"warning: {text} not defined")
                self.not_defined_text.add(text)
            text = 'n/a'
        return self.text_mapping[text]


nlp = spacy.load("en_core_web_sm")
llm_model = "meta-llama/Meta-Llama-3-8B-Instruct"

selfcheck_prompt = SelfCheckLLMPrompt(llm_model)
# selfcheck_prompt.model.generation_config.temperature=None
# selfcheck_prompt.model.generation_config.top_p=None


# LLM's text (e.g. GPT-3 response) to be evaluated at the sentence level  & Split it into sentences
passage = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Nation."
sentences = [sent.text.strip() for sent in nlp(passage).sents] # spacy sentence tokenization
print(sentences)
['Michael Alan Weiner (born March 31, 1942) is an American radio host.', 'He is the host of The Savage Nation.']

# Other samples generated by the same LLM to perform self-check for consistency
sample1 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Country."
sample2 = "Michael Alan Weiner (born January 13, 1960) is a Canadian radio host. He works at The New York Times."
sample3 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He obtained his PhD from MIT."

sent_scores_prompt = selfcheck_prompt.predict(
    sentences = sentences,                          # list of sentences
    sampled_passages = [sample1, sample2, sample3], # list of sampled passages
    verbose = True, # whether to show a progress bar
)
print(sent_scores_prompt)
# [0.33333333, 0.66666667] -- based on the example above