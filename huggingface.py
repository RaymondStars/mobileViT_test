import os
import torch
from torchvision import datasets
from transformers import MobileViTImageProcessor, MobileViTForImageClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

class MobileViTCollate:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        images = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs["labels"] = labels
        return inputs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    model_name = "apple/mobilevit-xx-small"
    print(f"Loaded model: {model_name}...")

    # Resize(288) -> CenterCrop(256) -> ToBGR -> Normalize
    processor = MobileViTImageProcessor.from_pretrained(model_name)
    model = MobileViTForImageClassification.from_pretrained(model_name).to(device)
    model.eval()  

    data_dir = "./imagenet_val"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Cannot find directory: {data_dir}, please ensure prepare_dataset.py is runned.")
        
    print(f"Reading dataset in: {data_dir} ...")
    val_dataset = datasets.ImageFolder(root=data_dir)
    
    custom_collate = MobileViTCollate(processor)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=custom_collate,
        pin_memory=True if torch.cuda.is_available() else False
    )

    correct_top1 = 0
    total = 0

    print("Evaluting MobileViT with ImageNet-1k validation set:")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating MobileViT"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values)
            logits = outputs.logits
            
            preds = logits.argmax(dim=-1)
            correct_top1 += (preds == labels).sum().item()
            total += labels.size(0)

    top1_accuracy = (correct_top1 / total) * 100
    print("\n" + "="*40)
    print(f" Evaluation Results for MobileViT on ImageNet-1k Validation Set:")
    print(f" Model Name: {model_name}")
    print(f" Total Images Evaluated: {total}")
    print(f" Top-1 Accuracy: {top1_accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
