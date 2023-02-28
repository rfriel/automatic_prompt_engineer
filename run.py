import json, time
import automatic_prompt_engineer.ape
from automatic_prompt_engineer.flan import *
import automatic_prompt_engineer.flan
import automatic_prompt_engineer.flan_singleton

def generate_text(self, prompts, n):
    for i in range(len(prompts)):
        p = prompts[i]
        if '[APE]' in p[:-len('[APE]')]:
            raise ValueError(p)
        prompts[i] = p.replace('[APE]', '')
    print(f"FlanForward received n={n} and prompts\n\t{prompts}")
    outs = []
    for _ in range(n):
        for i in range(0, len(prompts), self.bs):
            batch = prompts[i:i + self.bs]
            outs.extend(call_flan_t5(self.model, self.tokenizer, batch))
    return outs

def log_probs(self, text, output, neg_outputs=None, log_prob_range=None, debug=False):
    if not isinstance(text, list):
        text = [text]
    text = [s.lstrip() for s in text]
    # print(f"FlanForward log_probs received text={text} and output\n\t{output}")
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

    logits_out = []

    for i in range(logits.shape[0]):
        logits_out.append(
            logits[i, np.arange(logits.shape[1] - 1), output_tokens[i, 1:]]
        )
    # logits_out = logits[
    #     np.arange(logits.shape[0]), np.arange(logits.shape[1] - 1), output_tokens[:, 1:]].tolist()

    return logits_out, [[self.tokenizer.decode([c]) for c in ot[1:]] for ot in
                                    output_tokens]

automatic_prompt_engineer.flan.FlanForward.generate_text = generate_text
automatic_prompt_engineer.flan.FlanForward.log_probs =log_probs
automatic_prompt_engineer.flan.FlanForward._log_probs = _log_probs


def callback_fn(res, rounds, current_step, done=False):
    # prompts, scores = res.sorted()
    prompts, scores = res.prompts, res.scores
    scores = list(map(float, scores))

    prompts_, scores_ = res.sorted()
    out = dict(
        best_prompt=dict(prompt=prompts_[0], score=scores_[0]),
        all_prompts=[
            dict(prompt=p, score=s) for p, s in zip(prompts, scores)
            if s > -900
        ]
    )

    out['done'] = done
    out['num_steps'] = rounds
    out['current_step'] = current_step

    with open('output.json', 'w') as f:
        json.dump(out, f)



def defaults(num_prompts=40):
    eval_template = \
        """[PROMPT]
    
        [full_DEMO]
    
        [INPUT]
    
        [OUTPUT]"""

    demos_template = '''
    [INPUT]

    [OUTPUT]'''

    prompt_gen_template = """
    This is a list of inputs and outputs. What instruction is being followed?

    Input: 1

    Output: 2

    Input: 5

    Output: 6

    Input: 10

    Output: 11

    Input: 15

    Output: 16

    Instruction: Add 1 to the following number.

    This is a list of inputs and outputs. What instruction is being followed?

    [full_DEMO]

    Instruction: [APE]"""

    import automatic_prompt_engineer.config

    conf = automatic_prompt_engineer.config.simple_config(
        eval_model='',
        prompt_gen_model='text-davinci-002',
        prompt_gen_mode='forward',
        num_prompts=num_prompts,
        eval_rounds=num_prompts,
        prompt_gen_batch_size=100,
        eval_batch_size=100)

    conf['generation']['model']['gpt_config']['temperature'] = 1.0
    conf['generation']['model']['gpt_config']['top_p'] = 0.95
    conf['generation']['num_demos'] = 4
    conf['evaluation']['num_prompts_per_round'] = 1
    conf['rounds'] = num_prompts

    conf['evaluation']['callback_fn'] = callback_fn

    return eval_template, demos_template, prompt_gen_template, conf


def run(base_prompt, eval_data, num_prompts=40, max_eval=1600, seed=345433):
    eval_template, demos_template, prompt_gen_template, conf = defaults(num_prompts=num_prompts)

    conf['evaluation']['base_eval_config']['num_samples'] = min(len(eval_data[0]), max_eval)
    conf['evaluation']['base_eval_config']['sampling_seed'] = seed

    (res, eval_template, eval_data, demos_template, few_shot_data,
     config, current_step, num_steps), demo_fn = automatic_prompt_engineer.ape.find_prompts(
        eval_template,
        demos_template,
        eval_data,
        eval_data,
        conf,
        seed_prompts=[base_prompt],
        prompt_gen_template=prompt_gen_template,
        flan=True
    )
    return res, current_step, num_steps


def process_new_data(data):
    base_prompt = data['base_prompt']

    eval_data = ([], [])
    for row in data['labeled_examples']:
        eval_data[0].append(row['x'])
        eval_data[1].append(row['y'])

    num_prompts = data.get('num_prompts', 20)
    max_eval = data.get('max_eval', 32)
    seed = data.get('seed', 345433)

    res, current_step, num_steps = run(base_prompt, eval_data,
                                       num_prompts=num_prompts,
                                       max_eval=max_eval,
                                       seed=seed)

    # callback_fn(res, current_step, num_steps, done=True)


if __name__ == '__main__':
    while 1:
        with open('input.json') as f:
            data = json.load(f)

        if len(data) > 0:
            with open('input.json', 'w') as f:
                json.dump({}, f)

            process_new_data(data)
        time.sleep(0.5)