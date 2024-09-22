import os
from datasets import load_dataset
import fire
import numpy as np
import pandas as pd
import random


def build_human_summarization_dataset(data, prediction_criterion="random", seed=17):
    random.seed(seed)
    data_dict = {}

    for row in data:
        paper = data_dict.get(row["paper_id"], {})
        data_dict[row["paper_id"]] = paper
        paper["paper_id"] = row["paper_id"]
        paper["paper_name"] = row["paper_name"]
        paper["text"] = row["text"]
        summaries = paper.get("summaries", [])
        paper["summaries"] = summaries
        summaries.append(row["summary"])

    papers = []

    for _, paper in data_dict.items():
        if len(paper["summaries"]) < 2:
            continue
        summaries = paper["summaries"].copy()

        # The 'shorter' and 'longer' are designed to test
        # consistently longer or shorter human summaries.
        if prediction_criterion == "longer":
            sum_lengths = [len(s.split()) for s in summaries]
            prediction = summaries[np.argmax(sum_lengths)]
            summary = summaries[np.argmin(sum_lengths)]
        elif prediction_criterion == "shorter":
            sum_lengths = [len(s.split()) for s in summaries]
            prediction = summaries[np.argmin(sum_lengths)]
            summary = summaries[np.argmax(sum_lengths)]
        else:
            random.shuffle(summaries)
            summary = summaries[0]
            prediction = summaries[1]

        papers.append(
            dict(
                paper_id=paper["paper_id"],
                text=paper["text"],
                summary=summary,
                prediction=prediction,
            )
        )
    return papers


def run(prediction_criterion="random", split="validation", overwrite=False, **kwargs):
    save_to = f"data/mup_human_{prediction_criterion}_{split}.csv"
    
    if os.path.exists(save_to) and not overwrite:
        print(f"Dataset already exists: {save_to}")
        print("To overwrite the dataset, use '--overwrite'")
        mup_data = pd.read_csv(save_to)
        print("Loaded existing dataset:", save_to)
        new_dataset = False
    else:
        mup = load_dataset("allenai/mup")[split]
        assert len(mup) == 3604, "The validation split should have 3604 samples!"
        mup_data = build_human_summarization_dataset(mup, prediction_criterion, **kwargs)
        mup_data = pd.DataFrame(mup_data)
        mup_data.to_csv(save_to)
        print("Saved MuP summarization dataset:", save_to)
        new_dataset = True
    
    avg_summary_len = mup_data.summary.apply(lambda x: len(x.split())).mean()
    avg_prediction_len = mup_data.prediction.apply(lambda x: len(x.split())).mean()
    print("Average reference summary length:", avg_summary_len)
    print("Average human summary length:", avg_prediction_len)
    if not new_dataset and prediction_criterion == "random":
        assert round(avg_summary_len, 2) == 100.22
        assert round(avg_prediction_len, 2) == 99.52
    

if __name__ == "__main__":
    fire.Fire(run)
