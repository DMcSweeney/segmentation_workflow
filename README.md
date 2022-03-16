Workflow for building segmentation models

__All configs in: `config.ini`__

1. Reads directory of .npy files in <ROOT_DIR>/slices & <ROOT_DIR>/masks.
2. Run `bash split_data.sh`
3. `python train.py` - defaults to GPU:0