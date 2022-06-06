"""
Script for splitting data in Train test and validation sets
adapted from TrainTestValSplit.py to use folders of .npy files instead of .npz files
Output structure: data -> training/validation/testing -> slices 
                                                      -> masks
"""
import numpy as np
import os
import shutil
import configparser
from argparse import ArgumentParser

#~ Arg. Parse for config file
parser = ArgumentParser(description='Inference for testing segmentation model')
parser.add_argument('--config', '--c', dest='config',
                    default='config.ini', type=str, help='Path to config.ini file')
args = parser.parse_args()

#~ ====== Read config file =========
config = configparser.ConfigParser()
config.read(args.config)

root_dir = config['DIRECTORIES']['InputDirectory']
mask_path = os.path.join(root_dir, 'masks')
slices_path = os.path.join(root_dir, 'slices')

#* Output paths
split_output = os.path.join(root_dir, 'split_data')
os.makedirs(split_output, exist_ok=True)

train_path = os.path.join(split_output, 'training')
test_path = os.path.join(split_output, 'testing')
val_path = os.path.join(split_output, 'validation')

#*Train + (Test/Val)
train_test_split = float(config['TRAINTESTVAL']['TrainTestRatio'])

#* Test + Val
test_val_split = float(config['TRAINTESTVAL']['TestValRatio'])


def copy_files(names, out_path):
    #~ Copy files to destination
    print(f'Writing {len(names)} files to {out_path}')
    #* Make directories
    slice_dir = os.path.join(out_path, 'slices/')
    mask_dir = os.path.join(out_path, 'masks/')
    
    os.makedirs(slice_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for name in names:
        filename = name + '.npy'
        if filename in os.listdir(slice_dir):
            pass
        else:
            shutil.copyfile(os.path.join(slices_path, filename),
                            slice_dir + filename)
        # Check masks
        if filename in os.listdir(mask_dir):
            pass
        else:
            shutil.copyfile(os.path.join(mask_path, filename),
                            mask_dir + filename)

def main():
    #~ ====  Train/Test/Val Split =======
    ids = np.array([file.strip('.npy') for file in os.listdir(slices_path)])
    seed = int(config['TRAINTESTVAL']['Seed'])
    print('Shuffling...., seed:', seed)

    np.random.seed(seed=seed)
    indices = np.random.rand(len(ids)).argsort()
    #~ Data sizes
    train_size = int(len(ids)*train_test_split)
    test_size = int((len(ids)-train_size)*test_val_split)
    val_size = int((len(ids)-train_size)*(1-test_val_split))
    print('TRAIN/TEST/VAL', train_size, test_size, val_size)
    #~ Get IDs
    train_ids = ids[indices[:train_size]]
    val_ids = ids[indices[-val_size:]]
    test_ids = ids[indices[train_size:train_size+test_size]]

    #~ Copy files to destination
    copy_files(train_ids, train_path)
    copy_files(val_ids, val_path)
    copy_files(test_ids, test_path)


if __name__ == '__main__':
    main()
