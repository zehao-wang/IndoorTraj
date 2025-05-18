data_root=. # change this to your workspace root

sam_ratio=0.2
for base_dset in bonsai counter kitchen room; do
    for round in r1 r2 r3 r4 r5; do
        for stg in random uniform df dpp; do
            python main.py \
            --input-dir ${data_root}/data/processed/mipnerf360/mp-colmaps-${base_dset} \
            --out-dir ${data_root}/data/subsets/mipnerf360/${round}/${base_dset} \
            --cache-dir ${data_root}/data/processed/mipnerf360/${base_dset} \
            --sample_ratio ${sam_ratio} --sample_strategy ${stg} --measure jointm
        done
    done

    round=r1
    python main.py \
        --input-dir ${data_root}/data/processed/mipnerf360/mp-colmaps-${base_dset} \
        --out-dir ${data_root}/data/subsets/mipnerf360/${round}/${base_dset} \
        --cache-dir ${data_root}/data/processed/mipnerf360/${base_dset} \
        --sample_ratio ${sam_ratio} --sample_strategy cf --measure jointm
done 
