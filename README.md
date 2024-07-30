# S3F
Sequence-Structure-Surface Model for Protein Fitness Prediction (S3F)

## Dataset Process

```
# Download raw cath dataset
wget https://huggingface.co/datasets/tyang816/cath/blob/main/dompdb.tar -P ./dataset
tar -xvf dompdb.tar -C ./dataset
python script/preload_dataset.py -i ./dataset/dompdb/ -o ./dataset/processed_cath/

# or download the processed cath dataset
wget https://<link_to_processed_cath>/processed_cath.zip -P ./dataset
unzip ./dataset/processed_cath.zip -d ./dataset
```

```
# Process surface graphs (require one GPU to compute)
python script/process_surface.py -i ./dataset/processed_cath/ -o ./dataset/processed_surface_cath/

# or download the processed surface graphs
wget https://<link_to_processed_surface_cath>/processed_surface_cath.zip -P ./dataset
unzip ./dataset/processed_surface_cath.zip -d ./dataset
```

## Pre-train on CATH dataset with residue type prediction

The output files can be found at `~/scratch/proteingym_output`, which is specified by the `output_dir` argument in the `*.yaml`.

There is a `task.model.sequence_model.path` argument in each config file to control where to automatically download ESM model weights. Please modify this to your customized path to the esm model weights.

```
# Pre-train S2F model
python -m torch.distributed.launch --nproc_per_node=4 script/pretrain.py -c config/pretrain/s2f.yaml --datadir ./dataset/processed_cath

# Pre-train S3F model
python -m torch.distributed.launch --nproc_per_node=4 script/pretrain.py -c config/pretrain/s3f.yaml --datadir ./dataset/processed_cath --surfdir ./dataset/processed_surface_cath
```

# Evaluate on ProteinGym dataset

```
# Download ProteinGym benchmark
wget https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions.zip -O ./dataset
unzip DMS_ProteinGym_substitutions.zip -d ./dataset/DMS_ProteinGym_substitutions
wget https://marks.hms.harvard.edu/proteingym/ProteinGym_AF2_structures.zip -O ./dataset
unzip ProteinGym_AF2_structures.zip -d ./dataset/ProteinGym_AF2_structures

# Process surface graphs
python script/preload_dataset.py -i ./dataset/ProteinGym_AF2_structures/ -o ./dataset/processed_proteingym/
python script/process_surface.py -i ./dataset/processed_proteingym/ -o ./dataset/processed_surface_proteingym/
# or download the processed surface graphs
wget https://<link_to_processed_surface_proteingym>/processed_surface_proteingym.zip -P ./dataset
unzip ./dataset/processed_surface_proteingym.zip -d ./dataset
```

Only support single-gpu evaluation

```
# Run evaluation for S2F
python script/evaluate.py -c config/evaluate/s2f.yaml --datadir ./dataset/DMS_ProteinGym_substitutions --structdir ./dataset/ProteinGym_AF2_structures --ckpt <path_to_ckpt>

# Run evaluation for S3F
python script/evaluate.py -c config/evaluate/s3f.yaml --datadir ./dataset/DMS_ProteinGym_substitutions --structdir ./dataset/ProteinGym_AF2_structures --surfdir ./dataset/processed_surface_proteingym/ --ckpt <path_to_ckpt>
```

