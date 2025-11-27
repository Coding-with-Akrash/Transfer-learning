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

        print("Downloading skin cancer dataset...")
        path_skin = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
        print("Path to skin cancer dataset files:", path_skin)
        # Copy the HAM10000_images_part_1 and part_2 to local dataset TRAIN
        source_skin1 = os.path.join(path_skin, 'HAM10000_images_part_1')
        source_skin2 = os.path.join(path_skin, 'HAM10000_images_part_2')
        skin_train_path = os.path.join(train_path, 'SKIN_CANCER')
        os.makedirs(skin_train_path, exist_ok=True)
        for src in [source_skin1, source_skin2]:
            if os.path.exists(src):
                for file in os.listdir(src):
                    shutil.copy(os.path.join(src, file), os.path.join(skin_train_path, file))
        # Note: For simplicity, copying images without labels; in real scenario, need metadata.csv for labels
        print("Skin cancer dataset added to local directory.")

        print("Downloading brain tumor dataset...")
        path_brain = kagglehub.dataset_download("ahmedhamada0/brain-tumor-detection")
        print("Path to brain tumor dataset files:", path_brain)
        # Copy the yes and no folders to local dataset TRAIN
        source_brain_yes = os.path.join(path_brain, 'yes')
        source_brain_no = os.path.join(path_brain, 'no')
        brain_train_path = os.path.join(train_path, 'BRAIN_TUMOR')
        os.makedirs(brain_train_path, exist_ok=True)
        if os.path.exists(source_brain_yes):
            shutil.copytree(source_brain_yes, os.path.join(brain_train_path, 'TUMOR'))
        if os.path.exists(source_brain_no):
            shutil.copytree(source_brain_no, os.path.join(brain_train_path, 'NO_TUMOR'))
        print("Brain tumor dataset added to local directory.")
    else:
        print("Dataset already exists locally.")

    return train_path