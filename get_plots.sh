#!/bin/sh

models=("bsc_2b_4thepoch_hf" 
        "fourth_epoch_bsc_7b_restart_mix1_all_fineweb_from_mix1_lr3e-5_lr3e-6_step68625_hf"
        "salamandra2b_ca-en-es_v0.2" 
        "salamandra7b_ca-en-es_v0.2")

# Run plot for single models
for model in "${models[@]}"; do
    python3 plot_scores_single_model.py -m "$model"
done

python3 plot_scores_multiple_models.py -m bsc_2b_4thepoch_hf,salamandra2b_ca-en-es_v0.2 -t "salamandra_2b" -n base,ins
python3 plot_scores_multiple_models.py -m fourth_epoch_bsc_7b_restart_mix1_all_fineweb_from_mix1_lr3e-5_lr3e-6_step68625_hf,salamandra7b_ca-en-es_v0.2 -t "salamandra_7b" -n base,ins
python3 plot_scores_multiple_models.py -m bsc_2b_4thepoch_hf,fourth_epoch_bsc_7b_restart_mix1_all_fineweb_from_mix1_lr3e-5_lr3e-6_step68625_hf -t "salamandra_base" -n 2b,7b
python3 plot_scores_multiple_models.py -m salamandra2b_ca-en-es_v0.2,salamandra7b_ca-en-es_v0.2 -t "salamandra_instructed" -n 2b,7b
python3 plot_scores_multiple_models.py -m bsc_2b_4thepoch_hf,fourth_epoch_bsc_7b_restart_mix1_all_fineweb_from_mix1_lr3e-5_lr3e-6_step68625_hf,salamandra2b_ca-en-es_v0.2,salamandra7b_ca-en-es_v0.2 -t "salamandra_all" -n 2b,7b,2bins,7bins