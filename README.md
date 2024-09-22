# Scientific Summarization with LLMs

Resources for the paper [Can Large Language Model Summarizers Adapt to Diverse Scientific Communication Goals?](https://aclanthology.org/2024.findings-acl.508/) (ACL 2024).

## Requirements
It is recommended to setup a Python 3.12 environment. Using [Miniconda](https://docs.anaconda.com/miniconda/), the environment is created as follows:
```
conda create -n scisum python=3.12
```

Then, activate the environment, clone this repository, and install the dependencies:
```
conda activate scisum
git clone https://github.com/thefonseca/scisum.git
cd scisum
pip install -r requirements.txt
```

## Abstract generation experiments (arXiv and PubMed)

The following commands evaluate Llama-2 and OpenAI models on abstract generation for [arXiv papers](https://huggingface.co/datasets/armanc/scientific_papers) (Section 4.2 of the paper). The evaluation includes four types of prompt: 
- Baseline (without guidance)
- Target conciseness (fixed sentence budget)
- Target conciseness + first person narrative
- Target conciseness + first person narrative + target keyword coverage

To evaluate abstract generation with Llama-2 use the `run.py` script:
```bash
python run.py --dataset arxiv --model llama-2-7b-chat --model_checkpoint_path /path/to/llama2/checkpoint --keyword_model factorsum
```

By default, logs, metrics, and predictions are written to the `./output` folder. These outputs can be disabled by setting `--output_dir None`. To perform the same experiments on the PubMed dataset (Appendix D), use `--dataset pubmed`. For the Llama-2 with classifier-free guidance (CFG), use the `--guidance_scale` and `--negative_prompt` parameters:
```bash
python run.py --dataset arxiv --model llama-2-7b-chat --model_checkpoint_path /path/to/llama2/checkpoint --keyword_model factorsum --guidance_scale 1.5 --negative_prompt "Write a summary of the article above."
```

The OpenAI model used in the paper, `gpt-3.5-turbo-0301`, is now deprecated. As an alternative, this command performs evaluation with `gpt-4o-mini`:
```bash
python run.py --dataset arxiv --model gpt-4o-mini --keyword_model factorsum
```

To evaluate on the 500 held-out arXiv samples (Table 5 in the paper): 
```bash
python run.py --dataset_name data/arxiv-cs_CL-202401.json --model gpt-4o-mini --max_samples 500 --keyword_model factorsum
```

## Lay summarization experiments (eLife)

The experiments with the [eLife dataset](https://huggingface.co/datasets/tomasg25/scientific_lay_summarisation) work in a similar way as the commands for abstract generation descrive above. 

```bash
# Llama-2
python run.py --dataset elife --model llama-2-7b-chat --model_checkpoint_path /path/to/llama2/checkpoint --keyword_model bart

# OpenAI
python run.py --dataset elife --model gpt-4o-mini --keyword_model bart

# Llama-2 with classifier-free guidance (CFG)
python run.py --dataset elife --model llama-2-7b-chat --model_checkpoint_path /path/to/llama2/checkpoint --keyword_model bart --guidance_scale 1.5 --negative_prompt "Write a summary of the article above."
```

For the experiments with varying sentence budgets (Figure 2):
```bash
# Llama-2
./budget_sweep_llama.sh

# OpenAI
./budget_sweep_openai.sh
```

## Review summarization experiments (MuP)

The dataset described in Section 4.1 is already provided in `data/mup_human_random_validation.csv`. To generate a new dataset based on [MuP](https://github.com/allenai/mup), use:
```
python mup.py --overwrite
```
Note that this new dataset might generate slighly different results compared to Table 1 in the paper. The evaluation commands are as follows:

```bash
# Human-written summaries
python run.py --model human --dataset mup

# Llama-2
python run.py --dataset mup --model llama-2-7b-chat --model_checkpoint_path /path/to/llama2/checkpoint

# OpenAI
python run.py --dataset mup --model gpt-4o-mini
```

## Citation
```
@inproceedings{fonseca-cohen-2024-large-language,
    title = "Can Large Language Model Summarizers Adapt to Diverse Scientific Communication Goals?",
    author = "Fonseca, Marcio  and
      Cohen, Shay",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.508",
    pages = "8599--8618",
}
```