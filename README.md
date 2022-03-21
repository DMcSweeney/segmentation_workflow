<h1> Workflow for building segmentation models </h1>

__All configs in: `config.ini`__

<h3> Training </h3>

1. Reads directory of .npy files in <ROOT_DIR>/slices & <ROOT_DIR>/masks.
2. Run `bash split_data.sh`
3. `python train.py` - defaults to `config.ini`. Specify different config file with `--c`.

<h3> Testing Inference </h3>

1. `python inference.py` - defaults to `config.ini`. Specify different config file with `--c`.