data_root=$(realpath ./) # change this to your workspace root
echo $data_root

cd gaussian-splatting

sam_ratio=0.05
global_steps=30001
dset_name=indoortraj
measure=jointm
for scene in kitchen_1 kitchen_2 openplan_1 living_1; do
    for round in r1 r2 r3 r4 r5; do
        for stg in random uniform df dpp; do
            input_dir=${data_root}/data/subsets/${dset_name}/${round}/${scene}_${stg}_${measure}_ratio${sam_ratio}
            if [ ! -d "$input_dir" ]; then
                echo "Directory $input_dir does not exist. Exiting."
                exit 1
            fi

            exp_name=${scene}-${stg}-${measure}-ratio${sam_ratio}
        
            dump_path=${data_root}/data/snap/gs/${dset_name}/${scene}/${round}/${exp_name}
            echo "Dumpping to ${dump_path}"
            mkdir -p ${dump_path}

            python train.py \
            -s ${input_dir} \
            -m ${dump_path} --eval \
            -w --resolution 2 --iterations ${global_steps} | tee ${dump_path}/out.log

            # python render.py -m ${dump_path} --skip_train
            # python metrics.py -m ${dump_path}

        done
    done
    
    # The CF strategy is deterministic, so we can run it separately
    round=r1
    stg=cf
    input_dir=${data_root}/data/subsets/${dset_name}/${round}/${scene}_${stg}_${measure}_ratio${sam_ratio}
    if [ ! -d "$input_dir" ]; then
        echo "Directory $input_dir does not exist. Exiting."
        exit 1
    fi

    exp_name=${scene}-${stg}-${measure}-ratio${sam_ratio}

    dump_path=${data_root}/data/snap/gs/${dset_name}/${scene}/${round}/${exp_name}
    echo "Dumpping to ${dump_path}"
    mkdir -p ${dump_path}

    python train.py \
    -s ${input_dir} \
    -m ${dump_path} --eval \
    -w --resolution 2 --iterations ${global_steps} | tee ${dump_path}/out.log

    # python render.py -m ${dump_path} --skip_train
    # python metrics.py -m ${dump_path}

done
