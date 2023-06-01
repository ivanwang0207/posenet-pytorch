import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import List
from tools import *


def write_txt_file(datasets: dict, data_path: str, mode: str) -> None:

    with open(data_path, 'w') as f:
        dir_name = "train_images-"
        f.write(f'{mode}ing dataset\n')
        f.write('ImageFile, Camera Position [X Y Z W P Q R]\n')
        f.write('\n')
        
        for id, dataset in datasets.items():
            full_dir_name = "".join([dir_name, str(id)])

            for name, pose in zip(dataset["filenames"], dataset["poses"]):
                full_file_name = os.path.join(full_dir_name, name)
                f.write(full_file_name + ' ')
                f.write(' '.join('{:0.15f}'.format(i) for i in pose) + '\n')

def extract_source_data(set_ids: List[int], csv_path: str) -> dict:
    
    try: 
        src_df = pd.read_csv(csv_path)
    except IOError as e:
        print(e)

    datasets = dict()
    for id in set_ids:
        datasets[id] = dict()
        slice_df = src_df[src_df["TrajectoryId"] == id]

        trans = slice_df.iloc[:, 3:6].to_numpy() / 1000 
        rot = slice_df.iloc[:, 6:].to_numpy()
        quats = euler_to_quaternion(rot[:, 0], rot[:, 1], rot[:, 2]) # w, x, y, z
        
        datasets[id]["filenames"] = slice_df.iloc[:, 0].to_list()
        datasets[id]["poses"] = np.column_stack((trans, quats)).tolist()
    
    return datasets


def compute_mean_image(src_data_root: str, tar_data_root: str, resize_shape: list,
                       save_resized_imgs: bool = True):

    imsize = resize_shape # (H, W)
    imlist =  np.loadtxt(os.path.join(tar_data_root, 'dataset_train.txt'),
                        dtype=str, delimiter=' ', skiprows=3, usecols=(0))
    mean_image = np.zeros((imsize[0], imsize[1], 3), dtype=np.float64)

    for i, impath in enumerate(imlist):
        print('[%d/%d]:%s' % (i+1, len(imlist), impath), end='\r')
        image = Image.open(os.path.join(src_data_root, impath)).convert('RGB')

        image = image.rotate(-90, expand=True) 
        image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
        mean_image += np.array(image).astype(np.float64)

        # save resized training images
        if save_resized_imgs:
            save_resized_path = os.path.join(tar_data_root, impath)
            save_root, _ = os.path.split(save_resized_path)
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            if not os.path.exists(save_resized_path):
                image.save(save_resized_path)
                print(f"{save_resized_path} save successfully!")
            else:
                print(f"{save_resized_path} already exists!")

    mean_image /= len(imlist)
    Image.fromarray(mean_image.astype(np.uint8)).save(os.path.join(tar_data_root, 'mean_image.png'))
    np.save(os.path.join(tar_data_root, 'mean_image.npy'), mean_image)

    # save resized test images
    imlist =  np.loadtxt(os.path.join(tar_data_root, 'dataset_test.txt'),
                        dtype=str, delimiter=' ', skiprows=3, usecols=(0))
    
    for i, impath in enumerate(imlist):
        print('[%d/%d]:%s' % (i+1, len(imlist), impath), end='\r')
        image = Image.open(os.path.join(src_data_root, impath)).convert('RGB')

        image = image.rotate(-90, expand=True) 
        image = image.resize((imsize[1], imsize[0]), Image.BICUBIC)
        
        # save resized testing images
        if save_resized_imgs:
            save_resized_path = os.path.join(tar_data_root, impath)
            save_root, _ = os.path.split(save_resized_path)
            if not os.path.exists(save_root):
                os.makedirs(save_root)

            if not os.path.exists(save_resized_path):
                image.save(save_resized_path)
                print(f"{save_resized_path} save successfully!")
            else:
                print(f"{save_resized_path} already exists!")


def main(src_data_root: str, tar_data_root: str, 
         train_set_ids: list, test_set_ids: list, resize_shape: list):
    
    if not os.path.exists(tar_data_root):
            os.makedirs(tar_data_root)
    
    src_csv_path = os.path.join(src_data_root, "train_labels.csv")
    train_txt_path = os.path.join(tar_data_root, "dataset_train.txt")
    test_txt_path = os.path.join(tar_data_root, "dataset_test.txt")

    if os.path.exists(src_csv_path):
        training_sets = extract_source_data(train_set_ids, src_csv_path)
        testing_sets = extract_source_data(test_set_ids, src_csv_path)
        write_txt_file(training_sets, train_txt_path, "train")
        write_txt_file(testing_sets, test_txt_path, "test")

    # compute_mean_image(src_data_root, tar_data_root, resize_shape)

if __name__ == "__main__":

    src_data_root = "/media/ivanwang/Backup/AISG"
    tar_data_root = "../datasets/AISG"
    train_set_ids = [1,2,3]
    test_set_ids = [4]
    resize_shape = [360, 256]

    main(src_data_root, tar_data_root, train_set_ids, test_set_ids, resize_shape)