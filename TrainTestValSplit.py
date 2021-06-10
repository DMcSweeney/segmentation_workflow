"""
Script for splitting data in Train test and validation sets
adapted from TrainTestValSplit.py to use folders of .npy files instead of .npz files
Output structure: data -> training/validation/testing -> slices -> coronal/sagittal
                                                        -> targets
"""
import numpy as np
import argparse
import sys
import os
import shutil

# Arguments
shuffle = True
seed = 66

root_dir = 'data'
mask_path = f'./{root_dir}/masks/'
slices_path = f'./{root_dir}/slices/'

out_root = 'data'
# output paths
train_path = f'./{out_root}/training/'
test_path = f'./{out_root}/testing/'
val_path = f'./{out_root}/validation/'

#Train + (Test/Val)
train_test_split = 0.7

# Test + Val
test_val_split = 1.0


def copy_files(names, out_path):
    print(f'Writing {len(names)} files to {out_path}')
    # Make directories
    os.makedirs(os.path.dirname(
        f'{out_path}slices/'), exist_ok=True)
    os.makedirs(os.path.dirname(
        f'{out_path}masks/'), exist_ok=True)

    for name in names:
        filename = name + '.npy'
        # Check if in coronal
        if filename in os.listdir(f'{out_path}slices/'):
            pass
        else:
            shutil.copyfile(slices_path + filename,
                            f'{out_path}slices/{filename}')
        # Check masks
        if filename in os.listdir(f'{out_path}masks/'):
            pass
        else:
            shutil.copyfile(mask_path + filename,
                            f'{out_path}masks/{filename}')



def main():
    ids = np.array([file.strip('.npy') for file in os.listdir(slices_path)])
    print('Shuffling...., seed:', seed)

    np.random.seed(seed=seed)
    indices = np.random.rand(len(ids)).argsort()
    # Data sizes
    train_size = int(len(ids)*train_test_split)
    test_size = int((len(ids)-train_size)*test_val_split)
    val_size = int((len(ids)-train_size)*(1-test_val_split))
    print('TRAIN/TEST/VAL', train_size, test_size, val_size)
    train_ids = ids[indices[:train_size]]
    val_ids = ids[indices[-val_size:]]
    test_ids = ids[indices[train_size:train_size+test_size]]
    copy_files(train_ids, train_path)
    copy_files(val_ids, val_path)
    copy_files(test_ids, test_path)


if __name__ == '__main__':
    main()
