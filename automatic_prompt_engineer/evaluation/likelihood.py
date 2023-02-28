from automatic_prompt_engineer import llm, data, evaluate
import numpy as np

special_output_token = '[[[[OUTPUT]]]]'


def get_query(prompt, eval_template, input_, output_, demo_data, demos_template):
    """
    Returns the text sent to the LLM for likelihood evaluation.
    Parameters:
        prompt: The prompt.
        eval_template: The template for the evaluation queries.
        input_: The input.
        output_: The output.
    Returns:
        The query for the LLM and the range of the output text in the form of (start_idx, end_idx).
    """
    demos = demos_template.fill(demo_data)
    query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output=output_,
                               full_demo=demos)
    query_without_output = eval_template.fill(prompt=prompt,
                                              input=input_,
                                              output=special_output_token,
                                              full_demo=demos)

    first_idx = query_without_output.find(special_output_token)
    output_idx = first_idx, first_idx + len(output_)
    return query, output_idx


def get_query_encdec(prompt, eval_template, input_, output_, demo_data, demos_template):
    """
    Returns the text sent to the LLM for likelihood evaluation.
    Parameters:
        prompt: The prompt.
        eval_template: The template for the evaluation queries.
        input_: The input.
        output_: The output.
    Returns:
        The query for the LLM and the range of the output text in the form of (start_idx, end_idx).
    """
    demos = demos_template.fill(demo_data)
    query_without_output = eval_template.fill(prompt=prompt,
                                              input=input_,
                                              output="",
                                              full_demo=demos)

    return query_without_output, None



def likelihood_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config, verbose=False,
                         logprob_fn=None,
                         get_query_fn=None):
    print('in likelihood_evaluator')
    """
    For each prompt, evaluate the likelihood of the data (output) given the prompt.
    Parameters:
        prompts: A list of prompts.
        eval_template: The template for the evaluation queries.
        eval_data: The data to use for evaluation.
        config: The configuration dictionary.
    Returns:
        A LikelihoodEvaluationResult object.
    """
    print(f"logprob_fn config['logprob_fn'] {config['logprob_fn']}")
    logprob_fn = logprob_fn or config.get('logprob_fn')
    get_query_fn = get_query_fn or config.get('get_query_fn')
    queries = []
    output_indices = []
    outputs = []
    for prompt in prompts:
        subsampled_data = data.subsample_data(
            eval_data, config['num_samples'])
        for d in zip(*subsampled_data):
            input_, output_ = d
            demo_data = data.subsample_data(
                few_shot_data, config['num_few_shot'])

            if verbose:
                print('---------')
                print(prompt)
                print(eval_template)
                print(input_)
                print(output_)
                print(demo_data)
                print()

            get_query_ = get_query_fn if get_query_fn else get_query
            query, output_idx = get_query_(
                prompt, eval_template, input_, output_, demo_data, demos_template)
            queries.append(query)
            output_indices.append(output_idx)
            outputs.append(output_)

            if verbose:
                print(query)

    # Instantiate the LLM
    if logprob_fn:
        log_probs = logprob_fn(queries, outputs)
    else:
        model = llm.model_from_config(config['model'])
        log_probs, _ = model.log_probs(queries, output_indices)

    res = LikelihoodEvaluationResult(prompts, log_probs, config['num_samples'])

    return res


class LikelihoodEvaluationResult(evaluate.EvaluationResult):
    """
    A class for storing the results of a likelihood evaluation. Supports
    sorting prompts by various statistics of the likelihoods.
    """

    def __init__(self, prompts, log_probs, num_samples):
        self.prompts = prompts
        self.log_probs = log_probs
        self.prompt_log_probs = self._compute_avg_likelihood(
            prompts, log_probs, num_samples)

    def _compute_avg_likelihood(self, prompts, log_probs, num_samples):
        i = 0
        prompt_log_probs = []
        for prompt in prompts:
            prompt_log_probs.append([])
            for _ in range(num_samples):
                lps = log_probs[i]
                prompt_log_probs[-1].append(sum(lps) / len(lps))
                i += 1
        return prompt_log_probs

    def _agg_likelihoods(self, method):
        """For each prompt, compute a statistic of the likelihoods (e.g., mean, median, etc.)"""
        if method == 'mean':
            return [np.mean(lps) for lps in self.prompt_log_probs]
        elif method == 'median':
            return [np.median(lps) for lps in self.prompt_log_probs]
        elif method == 'std':
            return [np.std(lps) for lps in self.prompt_log_probs]
        elif method == 'max':
            return [np.max(lps) for lps in self.prompt_log_probs]
        elif method == 'min':
            return [np.min(lps) for lps in self.prompt_log_probs]
        elif method == 'iqm':
            return [np.mean(np.percentile(lps, [25, 75])) for lps in self.prompt_log_probs]
        else:
            raise ValueError(
                f'Unknown method {method} for aggregating likelihoods')

    def sorted(self, method='default'):
        if method == 'default':
            scores = self._agg_likelihoods('mean')
        else:
            scores = self._agg_likelihoods(method)
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_likelihoods('mean')
        else:
            scores = self._agg_likelihoods(method)
        return self.prompts, scores

    def __str__(self):
        s = ''
        prompts, scores = self.sorted()
        s += 'log(p): prompt\n'
        s += '----------------\n'
        for prompt, score in list(zip(prompts, scores))[:10]:
            s += f'{score:.2f}: {prompt}\n'
        return s


def score_likelihood(candidates,
                     eval_template,
                     eval_data,
                     demos_template,
                     few_shot_data,
                     num_few_shot=5,
                     method='mean',
                     flan=True,
                     verbose=False):
    """for grips"""
    get_query_fn, logprob_fn = None, None
    if flan:
        import automatic_prompt_engineer.flan_singleton
        logprob_fn = automatic_prompt_engineer.flan_singleton.FLAN_APE.log_probs
        get_query_fn = get_query_encdec

    eval_config = dict(num_few_shot=num_few_shot, num_samples=len(eval_data[0]))

    out = likelihood_evaluator(
        candidates, eval_template, eval_data, demos_template, few_shot_data,
        eval_config,
        get_query_fn=get_query_fn,
        logprob_fn=logprob_fn,
        verbose=verbose,
        )
    return out._agg_likelihoods(method=method)