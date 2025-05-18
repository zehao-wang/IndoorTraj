# Preparing ScanNet Dataset

## Data Download

Please follow the instruction in [ScanNet](https://github.com/ScanNet/ScanNet) official repo for dataset download agreement. You will get file ```download-scannetv2.py``` from ScanNet team.

Please use the script to download the raw ScanNet

```bash
sh download_scannetv2.sh 0084_00
```

## Extract cam info and frames

```bash
sh process_scannet.sh 0084_00
```

## Transfer to Blender style data
Please change the scene list in the following script.

```bash
sh scannet2gsformat.sh
```

