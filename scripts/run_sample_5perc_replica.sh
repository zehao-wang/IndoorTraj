data_root=. # change this to your workspace root

sam_ratio=0.05
for base_dset in office2 office3 room1 room2; do
    for round in r1 r2 r3 r4 r5; do
        for stg in random uniform df dpp; do
            python main.py \
            --input-dir ${data_root}/data/processed/replica/replica_part_test/${base_dset} \
            --out-dir ${data_root}/data/subsets/replica/${round}/${base_dset} \
            --cache-dir ${data_root}/data/processed/replica/replica_part_test/${base_dset} \
            --sample_ratio ${sam_ratio} --sample_strategy ${stg} --measure jointm
        done
    done

    round=r1
    python main.py \
        --input-dir ${data_root}/data/processed/replica/replica_part_test/${base_dset} \
        --out-dir ${data_root}/data/subsets/replica/${round}/${base_dset} \
        --cache-dir ${data_root}/data/processed/replica/replica_part_test/${base_dset} \
        --sample_ratio ${sam_ratio} --sample_strategy cf --measure jointm
done 
