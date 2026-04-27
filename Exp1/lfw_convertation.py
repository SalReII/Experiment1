import os
import cv2
import albumentations as A
from tqdm import tqdm

input_path = "/Users/salrei/Documents/Diss/Exp1/Dataset/lfw_orig"
base_output = "/Users/salrei/Documents/Diss/Exp1/Dataset/lfw_new"

DEGRADATIONS = {
    'blur': [
        A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        A.GaussianBlur(blur_limit=(5, 5), p=1.0),
        A.GaussianBlur(blur_limit=(7, 7), p=1.0),
        A.GaussianBlur(blur_limit=(9, 9), p=1.0),
        A.GaussianBlur(blur_limit=(13, 15), p=1.0),
    ],
    'noise': [
        A.GaussNoise(std_range=(0.02, 0.04), p=1.0), 
        A.GaussNoise(std_range=(0.05, 0.1), p=1.0),
        A.GaussNoise(std_range=(0.15, 0.2), p=1.0),
        A.GaussNoise(std_range=(0.25, 0.35), p=1.0),
        A.GaussNoise(std_range=(0.4, 0.6), p=1.0),
    ],
    'low_res': [
        A.Downscale(scale_range=(0.5, 0.5), p=1.0), 
        A.Downscale(scale_range=(0.3, 0.3), p=1.0),
        A.Downscale(scale_range=(0.2, 0.2), p=1.0),
        A.Downscale(scale_range=(0.15, 0.15), p=1.0),
        A.Downscale(scale_range=(0.08, 0.08), p=1.0),
    ],
    'occlusion': [
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(20, 20), hole_width_range=(20, 20), fill=0, p=1.0),
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(35, 35), hole_width_range=(35, 35), fill=0, p=1.0),
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(50, 50), hole_width_range=(50, 50), fill=0, p=1.0),
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(65, 65), hole_width_range=(65, 65), fill=0, p=1.0),
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(85, 85), hole_width_range=(85, 85), fill=0, p=1.0),
    ]
}

all_files = []
for root, _, files in os.walk(input_path):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_files.append(os.path.join(root, f))

print(f"Найдено изображений: {len(all_files)}")

for deg_name, transforms in DEGRADATIONS.items():
    print(f"\nГенерация категории: {deg_name}")
    
    for lvl_idx, transform in enumerate(transforms, 1):
        for img_path in tqdm(all_files, desc=f"  Level {lvl_idx}", leave=False):
            rel_path = os.path.relpath(img_path, input_path)
            save_path = os.path.join(base_output, deg_name, f"level_{lvl_idx}", rel_path)
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            image = cv2.imread(img_path)
            if image is not None:
                augmented = transform(image=image)['image']
                cv2.imwrite(save_path, augmented)

print("Процесс завершен")