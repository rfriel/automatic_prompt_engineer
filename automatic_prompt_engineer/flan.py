import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from automatic_prompt_engineer import llm

FLAN_NAME = 'google/flan-t5-base'


def no_init(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


def load_flan(name=FLAN_NAME):
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained(name)

        model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16)
        model.cuda()
        model.requires_grad_(False)
        return model, tokenizer

    model, tokenizer = no_init(load_model)
    return model, tokenizer


def call_flan_t5(model, tokenizer, prompts, max_length=60, temperature=0.5, top_p=0.9, debug=False,
                 **kwargs):
    single_prompt = 0

    if isinstance(prompts, str):
        prompts = [prompts]
        single_prompt = 1

    ins = {k: v.cuda() for k, v in tokenizer(prompts, return_tensors='pt').items()}

    out = model.generate(**ins, do_sample=True, max_length=max_length, temperature=temperature,
                         top_p=top_p, **kwargs)

    if debug:
        return out

    out = tokenizer.batch_decode(out.cpu(), skip_special_tokens=True)

    if single_prompt:
        out = out[0]

    return out


class FlanForward(llm.LLM):
    def __init__(self, model, tokenizer, bs=1, disable_tqdm=False):
        self.model = model
        self.tokenizer = tokenizer
        self.bs = bs
        self.disable_tqdm = disable_tqdm

    @staticmethod
    def load(name='google/flan-t5-xl', bs=1, disable_tqdm=False):
        model, tokenizer = load_flan(name)
        return FlanForward(model, tokenizer, bs, disable_tqdm)

    def generate_text(self, prompts, n):
        outs = []
        for i in range(0, len(prompts), self.bs):
            batch = prompts[i:i + self.bs]
            outs.extend(call_flan_t5(self.model, self.tokenizer, batch))
        return outs

    def log_probs(self, text, output, log_prob_range=None, debug=False):
        if not isinstance(text, list):
            text = [text]
        batch_size = self.bs
        text_batches = [text[i:i + batch_size]
                        for i in range(0, len(text), batch_size)]
        output_batches = [output[i:i + batch_size]
                          for i in range(0, len(text), batch_size)]
        if log_prob_range is None:
            log_prob_range_batches = [None] * len(text)
        else:
            assert len(log_prob_range) == len(text)
            log_prob_range_batches = [log_prob_range[i:i + batch_size]
                                      for i in range(0, len(log_prob_range), batch_size)]
        if not self.disable_tqdm:
            print(
                f"Getting log probs for {len(text)} strings, "
                f"split into {len(text_batches)} batches of (maximum) size {batch_size}")
        log_probs = []
        tokens = []
        for text_batch, output_batch, log_prob_range in tqdm(
                list(zip(text_batches, output_batches, log_prob_range_batches)),
                disable=self.disable_tqdm):
            log_probs_batch, tokens_batch = self._log_probs(
                text_batch, output_batch, log_prob_range, debug=debug)
            log_probs += log_probs_batch
            tokens += tokens_batch
        # return log_probs, tokens
        return log_probs

    def _log_probs(self, prompt, output, log_prob_range=None, debug=False):
        ins = {k: v.cuda() for k, v in
               self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).items()}

        decoder_start_token = self.tokenizer.decode(
            [self.model.generation_config.decoder_start_token_id])
        output_tokens = \
        self.tokenizer([decoder_start_token + e for e in output], return_tensors='pt', padding=True,
                       truncation=True)['input_ids'][:, :-1].cuda()
        ins['decoder_input_ids'] = output_tokens

        outs = self.model(**ins)
        logits = outs.logits.cpu().numpy()
        output_tokens = output_tokens.cpu().numpy()

        if debug:
            return logits, ins['decoder_input_ids']

        logits_out = logits[
            np.arange(logits.shape[0]), np.arange(logits.shape[1] - 1), output_tokens[:, 1:]]

        return logits_out.tolist(), [[self.tokenizer.decode([c]) for c in ot[1:]] for ot in
                                     output_tokens]
