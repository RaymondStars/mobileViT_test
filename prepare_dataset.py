import os
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

print("Installing and loading the ImageNet validation dataset...")
ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

output_dir = "./imagenet_val"
os.makedirs(output_dir, exist_ok=True)

print("Installing and converting the dataset to local image files...")
for idx, item in enumerate(tqdm(ds)):
    img = item['image']
    label = item['label']
    
    # creating directory to directly use ImageFolder()
    class_dir = os.path.join(output_dir, f"{label:03d}")
    os.makedirs(class_dir, exist_ok=True)
    img_path = os.path.join(class_dir, f"val_{idx:05d}.jpg")
    
    # ensure the image is stored as RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')       
    img.save(img_path, "JPEG")

print(f"Completed! The dataset is saved in: {output_dir}")
