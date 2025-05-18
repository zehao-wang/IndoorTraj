data_root=. # change this to your workspace root

sam_ratio=0.05
for base_dset in kitchen_1 kitchen_2 openplan_1 living_1; do
    for round in r1 r2 r3 r4 r5; do
        for stg in random uniform df dpp; do
            python main.py \
            --input-dir ${data_root}/data/processed/indoortraj/${base_dset} \
            --out-dir ${data_root}/data/subsets/indoortraj/${round}/${base_dset} \
            --cache-dir ${data_root}/data/processed/indoortraj/${base_dset} \
            --sample_ratio ${sam_ratio} --sample_strategy ${stg} --measure jointm
        done
    done

    round=r1
    python main.py \
        --input-dir ${data_root}/data/processed/indoortraj/${base_dset} \
        --out-dir ${data_root}/data/subsets/indoortraj/${round}/${base_dset} \
        --cache-dir ${data_root}/data/processed/indoortraj/${base_dset} \
        --sample_ratio ${sam_ratio} --sample_strategy cf --measure jointm
done 
