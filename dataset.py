import os
import kagglehub
import shutil

def download_and_prepare_dataset():
    local_dataset_path = 'dataset'
    train_path = os.path.join(local_dataset_path, 'TRAIN')
    if not os.path.exists(train_path):
        print("Downloading blood cells dataset...")
        path_blood = kagglehub.dataset_download("paultimothymooney/blood-cells")
        print("Path to blood cells dataset files:", path_blood)
        # Copy the TRAIN folder to local dataset
        source_train_blood = os.path.join(path_blood, 'dataset2-master', 'dataset2-master', 'images', 'TRAIN')
        shutil.copytree(source_train_blood, train_path)
        print("Blood cells dataset copied to local directory.")

        print("Downloading chest X-ray dataset...")
        path_xray = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print("Path to chest X-ray dataset files:", path_xray)
        # Copy the TRAIN folder contents to local dataset TRAIN
        source_train_xray = os.path.join(path_xray, 'chest_xray', 'chest_xray', 'train')
        for item in os.listdir(source_train_xray):
            s = os.path.join(source_train_xray, item)
            d = os.path.join(train_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
        print("Chest X-ray dataset added to local directory.")
    else:
        print("Dataset already exists locally.")

    return train_path