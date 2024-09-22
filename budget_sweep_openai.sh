#!/bin/bash

if test -z "${DATASET}"; then
  DATASET='tomasg25/scientific_lay_summarisation'
fi
if test -z "${DATASET_CONFIG}"; then
  DATASET_CONFIG='elife'
fi
if test -z "${TARGET_KEY}"; then
  if [ "$DATASET_CONFIG" = 'elife' ]; then
    TARGET_KEY='summary'
  else
    TARGET_KEY='abstract'
  fi
fi
if test -z "${SPLIT}"; then
  SPLIT='validation'
fi
if test -z "${MAX_LENGTH}"; then
  MAX_LENGTH=1024
fi
if test -z "${MAX_SAMPLES}"; then
  MAX_SAMPLES=100
fi
if test -z "${SEED}"; then
  SEED=17
fi

for i in 1 2 4 6 8 10 12 14 16;
do
python evaluation.py \
--dataset_name $DATASET \
--dataset_config $DATASET_CONFIG --split $SPLIT \
--target_key $TARGET_KEY \
--model_name gpt-4o-mini \
--run_id gpt-4o-mini_budget_${i} \
--max_samples $MAX_SAMPLES \
--shuffle --seed $SEED \
--budget $i \
--max_length $MAX_LENGTH \
--ignore_errors \
--output_dir output
done
