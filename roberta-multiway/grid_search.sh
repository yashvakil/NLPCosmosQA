#!/bin/bash
conda activate
python --version

task="commonsenseqa"

batchsizes=( 24 )
for s in "${batchsizes[@]}"
do
    learningrates=( 1e-5 )

    for l in "${learningrates[@]}"
    do
        epochs=( 1 )

        for e in "${epochs[@]}"
        do
            python run_multiway_att.py --task_name "${task}" --do_eval --do_train --do_lower_case --bert_model bert-large-uncased --data_dir data/ --max_seq_length 220 --train_batch_size ${s} --learning_rate ${l} --num_train_epochs ${e}  --gradient_accumulation_steps=4  --output_dir output/batch_${s}_lr_${l}_epochs${e}_seed_7 --seed 7 --fp16
        done
    done
done
