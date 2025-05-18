dataset_dir='../../data/raw/MiP-NeRF360'
output_dir='../../data/processed/mipnerf360'

python process_mipnerf360.py \
--input ${dataset_dir} \
--output ${output_dir} \
--scenes room counter kitchen bonsai

