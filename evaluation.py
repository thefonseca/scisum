import logging
import os

import fire

from llms.summarizers.evaluation import evaluate_summarizer
from llms.metrics import rouge_score
from llms.utils.utils import config_logging, sent_tokenize

logger = logging.getLogger(__name__)


def control_metrics(
    prediction,
    reference=None,
    source=None,
    index=None,
    parallelized=None,
    keywords=None,
    **kwargs,
):
    metrics = {
        "keywords_prediction": {},
        "keywords_reference": {},
        "narrative": {},
    }

    def first_person_fraction(x):
        sents = sent_tokenize(x)
        sents = [s.lower().strip() for s in sents]
        fp_sents = [
            s
            for s in sents
            for p in ["we", "our"]
            if s.split(" ")[0] == p or f", {p} " in s
        ]
        return len(fp_sents) / len(sents)

    metrics["narrative"]["first_person_prediction"] = first_person_fraction(prediction)
    metrics["narrative"]["first_person_reference"] = first_person_fraction(reference)
    metrics["narrative"]["third_person_prediction"] = (
        1 - metrics["narrative"]["first_person_prediction"]
    )
    metrics["narrative"]["third_person_reference"] = (
        1 - metrics["narrative"]["first_person_reference"]
    )

    if keywords:
        if isinstance(keywords, list):
            keywords = keywords[index]
        if keywords is None or str(keywords) == "nan":
            keywords = ""

        score = rouge_score(prediction, keywords, rouge_ngrams=["rouge1"])
        metrics["keywords_prediction"]["rouge"] = score
        score = rouge_score(reference, keywords, rouge_ngrams=["rouge1"])
        metrics["keywords_reference"]["rouge"] = score

    return metrics


def evaluate(
    dataset_name=None,
    dataset_config=None,
    split=None,
    arxiv_id=None,
    arxiv_query=None,
    source_key="article",
    target_key="abstract",
    model_name=None,
    model_class=None,
    prediction_path=None,
    prediction_key="prediction",
    max_samples=None,
    output_dir=None,
    cache_start=0,
    cache_end=None,
    run_id=None,
    **kwargs,
):
    if arxiv_id or arxiv_query:
        dataset_name = "arxiv-api"

    timestr = config_logging(
        dataset_name, dataset_config, split, output_dir, run_id=run_id
    )
    metrics = [control_metrics]
    
    evaluate_summarizer(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        arxiv_id=arxiv_id,
        arxiv_query=arxiv_query,
        source_key=source_key,
        target_key=target_key,
        model_name=model_name,
        model_class=model_class,
        prediction_path=prediction_path,
        prediction_key=prediction_key,
        max_samples=max_samples,
        output_dir=output_dir,
        cache_start=cache_start,
        cache_end=cache_end,
        metrics=metrics,
        timestr=timestr,
        run_id=run_id,
        **kwargs,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate)
