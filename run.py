import os
from pathlib import Path

import fire

from evaluation import evaluate


def get_prompts(model, dataset, budget, keyword_model, max_keywords):
    if model is None:
        prompts = [None]
        run_ids = [""]
    elif any([x in model for x in ["bart", "bigbird"]]) or dataset == "mup":
        prompts = [None]
        run_ids = [model]
    else:
        prompt_no_budget = "Write a summary of the article above."
        prompt_budget = "Write a summary of the article above in {budget} sentences."
        prompt_budget_first_person = 'Write a summary of the article above in {budget} sentences. Write in first person plural "we" when applicable.'
        prompt_budget_third_person = "Write a summary of the article above in {budget} sentences. Write in third person."
        run_id_no_budget = f"{model}_no_budget"
        run_id_budget = f"{model}_budget_{budget}"
        run_id_budget_first_person = f"{model}_budget_{budget}_first_person"
        run_id_budget_third_person = f"{model}_budget_{budget}_third_person"

        if dataset in ["arxiv", "pubmed"]:
            prompts = [prompt_no_budget, prompt_budget, prompt_budget_first_person]
            run_ids = [run_id_no_budget, run_id_budget, run_id_budget_first_person]
        else:
            prompts = [prompt_no_budget, prompt_budget, prompt_budget_third_person]
            run_ids = [run_id_no_budget, run_id_budget, run_id_budget_third_person]
    
    if keyword_model is not None:
        first_person = dataset in ["arxiv", "pubmed"]
        kw_prompt = "Write a summary of the article above in {budget} sentences."
        kw_prompt += " Focus on the following keywords: {keywords}."
        if first_person:
            kw_prompt += ' Write in first person plural "we" when applicable.'

        for kw_type in ["pred_keywords"]:  # , "source_keywords"]:
            prompts.append(kw_prompt)
            if kw_type == "source_keywords":
                run_id = f"{model}_{kw_type}"
            else:
                run_id = f"{model}_{kw_type}_{keyword_model}"
            
            if max_keywords:
                run_id += f"{max_keywords}"
            run_ids.append(run_id)

    return prompts, run_ids


def run(
    dataset_name=None,
    dataset="arxiv",
    split="test",
    model=None,
    use_model_cache=True,
    budget=None,
    prompt=None,
    run_id=None,
    keyword_model=None,
    max_keywords=None,
    max_length=None,
    max_samples=1000,
    output_dir="output",
    seed=17,
    **kwargs,
):
    LOG_LEVEL_FINE = 15
    if "verbose" not in kwargs and output_dir is not None:
        os.environ["LOG_LEVEL"] = str(LOG_LEVEL_FINE)
    elif kwargs.pop("verbose", False) is True:
        os.environ["LOG_LEVEL"] = str(LOG_LEVEL_FINE)

    if dataset_name is None:
        if dataset in ["arxiv", "pubmed"]:
            dataset_name = "scientific_papers"
        elif dataset in ["elife", "plos"]:
            dataset_name = "tomasg25/scientific_lay_summarisation"
        elif dataset == "mup":
            dataset_name = "data/mup_human_random_validation.csv"

    if "prediction_path" not in kwargs:
        if dataset == "mup" and model == "human":
            kwargs["prediction_path"] = dataset_name
        else:
            kwargs["model_name"] = model

    if max_length is None:
        if model == "bart-base-elife":
            max_length = 1024
        elif dataset in ["pubmed", "elife"]:
            max_length = 512
        else:
            max_length = 256

    if budget is None:
        if dataset == "pubmed":
            kwargs["prompt_budget"] = 8
        elif dataset == "elife":
            kwargs["prompt_budget"] = 14
        elif dataset == "mup" and model == "llama-2-7b-chat":
            kwargs["prompt_budget_path"] = "data/mup_budgets_llama2.csv"
        elif dataset == "mup" and model == "gpt-3.5-turbo-0301":
            kwargs["prompt_budget_path"] = "data/mup_budgets_gpt-3.5.csv"
        elif dataset == "mup":
            kwargs["prompt_budget"] = 5
        else:
            kwargs["prompt_budget"] = 6
        budget = kwargs.get("prompt_budget")

    if "source_key" not in kwargs:
        if dataset == "mup":
            kwargs["source_key"] = "text"
        else:
            kwargs["source_key"] = "article"
    
    if "target_key" not in kwargs:
        if dataset in ["elife", "plos", "mup"]:
            kwargs["target_key"] = "summary"
        else:
            kwargs["target_key"] = "abstract"

    if "model_checkpoint_path" not in kwargs:
        if model == "llama-2-7b-chat":
            kwargs["model_checkpoint_path"] = "models/llama2/hf/7b-chat/"
        elif model == "bart-base-elife":
            kwargs["model_checkpoint_path"] = "artifacts/bart-base-elife"

    if "model_dtype" not in kwargs:
        if model == "llama-2-7b-chat":
            kwargs["model_dtype"] = "float16"

    if "ignore_errors" not in kwargs:
        if model and any([x in model for x in ["gpt-3.5", "gpt-4"]]):
            kwargs["ignore_errors"] = True

    if "model_request_interval" not in kwargs:
        if model and "gpt-3.5" in model:
            kwargs["model_request_interval"] = 2

    if "shuffle" not in kwargs:
        kwargs["shuffle"] = dataset not in ["elife", "mup"]

    run_ids = None
    if prompt is None:
        prompts, run_ids = get_prompts(model, dataset, budget, keyword_model, max_keywords)
    elif isinstance(prompt, str):
        prompts = [prompt]
    else:
        prompts = prompt

    if run_ids is None and isinstance(run_id, str):
        run_ids = [run_id]
    elif run_ids is None:
        run_ids = run_id

    print("INFO:evaluation.py: Using prompts:")
    for p in prompts:
        print(f"- {p}")

    def get_kw_path(kw_type, kw_model):
        path = f"data/{kw_type}_{dataset}_{split}_{kw_model}"
        if max_keywords is not None:
            path += f"_{max_keywords}"
        path += ".csv"
        return path

    for run_id, prompt in zip(run_ids, prompts):
        if prompt and Path(str(prompt)).exists():
            kwargs["user_prompt_path"] = prompt
        elif prompt:
            kwargs["model_user_prompt"] = prompt

        if "source_keywords" in run_id:
            kwargs["prompt_keywords_path"] = get_kw_path(
                "source_keywords", keyword_model
            )
        elif "pred_keywords" in run_id:
            kwargs["prompt_keywords_path"] = get_kw_path("pred_keywords", keyword_model)
        elif dataset == "elife":
            kwargs["prompt_keywords_path"] = get_kw_path("pred_keywords", "bart")
        elif dataset in ["arxiv", "pubmed"]:
            kwargs["prompt_keywords_path"] = get_kw_path("pred_keywords", "factorsum")

        evaluate(
            dataset_name=dataset_name,
            dataset_config=dataset,
            split=split,
            use_model_cache=use_model_cache,
            run_id=run_id,
            max_length=max_length,
            max_samples=max_samples,
            seed=seed,
            **kwargs,
        )


if __name__ == "__main__":
    fire.Fire(run)
