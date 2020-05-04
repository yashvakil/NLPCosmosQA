#!/bin/bash
source activate

task="commonsenseqa"

batchsizes=( 60 )
for s in "${batchsizes[@]}"
do
    learningrates=( 1e-5 )

    for l in "${learningrates[@]}"
    do
        epochs=( 2 )

        for e in "${epochs[@]}"
        do
            python run_roberta.py --task_name "${task}" --do_eval --do_train --load_model --do_lower_case --roberta_model roberta-large --data_dir data/ --max_seq_length 180 --train_batch_size ${s} --learning_rate ${l} --num_train_epochs ${e}  --gradient_accumulation_steps=10  --output_dir output/batch_${s}_lr_${l}_epochs${e}_seed_7 --seed 7 --fp16
        done
    done
done

#python run_roberta.py --task_name "commonsenseqa" --do_eval --do_train --load_model --do_lower_case --roberta_model roberta-large --data_dir data/ --max_seq_length 180 --train_batch_size 2 --learning_rate 1e-5 --num_train_epochs 1  --gradient_accumulation_steps=10  --output_dir output/batch_64_lr_1e-5_epochs1_seed_7 --seed 7 