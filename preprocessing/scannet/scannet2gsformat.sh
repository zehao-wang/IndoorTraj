dataset_dir='../../data/raw/ScanNet'
output_dir='../../data/processed/ScanNet'


python scannet2gsformat.py \
--input ${dataset_dir} \
--output ${output_dir} \
--scenes scene0050_00 scene0085_00 scene0134_02 scene0073_01 

