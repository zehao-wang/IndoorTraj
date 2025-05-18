data_root=. # change this to your workspace root

sam_ratio=0.05
for base_dset in scene0050_00 scene0073_01 scene0085_00 scene0134_02; do
    for round in r1 r2 r3 r4 r5; do
        for stg in random uniform df dpp; do
            python main.py \
            --input-dir ${data_root}/data/processed/ScanNet/scannet-${base_dset} \
            --out-dir ${data_root}/data/subsets/scannet/${round}/${base_dset} \
            --cache-dir ${data_root}/data/processed/ScanNet/${base_dset} \
            --sample_ratio ${sam_ratio} --sample_strategy ${stg} --measure jointm
        done
    done

    round=r1
    python main.py \
        --input-dir ${data_root}/data/processed/ScanNet/scannet-${base_dset} \
        --out-dir ${data_root}/data/subsets/scannet/${round}/${base_dset} \
        --cache-dir ${data_root}/data/processed/ScanNet/${base_dset} \
        --sample_ratio ${sam_ratio} --sample_strategy cf --measure jointm
done 
