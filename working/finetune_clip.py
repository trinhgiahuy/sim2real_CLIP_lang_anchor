# src/finetune_clip.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import os
from . import config, dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    print(f"Loading CLIP model: {config.CLIP_MODEL_NAME}")

    model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)


    # --- Prepare Data ---
    train_full_h5_path = os.path.join(config.PROCESSED_DATA_DIR, 'train_frames.h5') # Using train split for training
    val_full_h5_path = os.path.join(config.PROCESSED_DATA_DIR, 'val_frames.h5') # Using val split for validation
    stats_path = os.path.join(config.PROCESSED_DATA_DIR, "norm_stats.json")



    # The full dataset is loaded here
    train_dataset = dataset.RadarFrameDataset(train_full_h5_path, stats_path, processor)
    val_dataset = dataset.RadarFrameDataset(val_full_h5_path, stats_path, processor)



    # Split into training and validation sets
    # val_size = int(0.2 * len(full_dataset))

    # train_size = len(full_dataset) - val_size

    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])



    print(f"Total training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=False, num_workers=4)

    text_prompts = [config.TEXT_DESCRIPTORS[i] for i in range(len(config.CLASS_NAMES))]

    optimizer = torch.optim.Adam(model.parameters(), lr=config.FINETUNE_LR)



    # --- Early Stopping and Validation Logic ---
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting CLIP fine-tuning for {config.FINETUNE_EPOCHS} epochs...")

    for epoch in range(config.FINETUNE_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.FINETUNE_EPOCHS}"):
            pixel_values = batch['pixel_values'].to(device)
            ground_truth = batch['label'].to(device)
            text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
            outputs = model(pixel_values=pixel_values, **text_inputs)
            logits_per_image = outputs.logits_per_image
            loss = nn.functional.cross_entropy(logits_per_image, ground_truth)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)



        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.FINETUNE_EPOCHS} [Validation]"):
                pixel_values = batch['pixel_values'].to(device)
                ground_truth = batch['label'].to(device)
                text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
                outputs = model(pixel_values=pixel_values, **text_inputs)
                loss = nn.functional.cross_entropy(outputs.logits_per_image, ground_truth)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")



        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(config.FINETUNED_MODEL_PATH)
            processor.save_pretrained(config.FINETUNED_MODEL_PATH)
            print(f"Validation loss improved. Saved new best model.")
        else:
            patience_counter += 1
            print(f"INFO: No improvement in validation loss for {patience_counter} epoch(s).")


        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered after {patience_counter} epochs.")
            break

    model.save_pretrained(config.FINETUNED_MODEL_PATH)
    processor.save_pretrained(config.FINETUNED_MODEL_PATH)
    print(f"✅ Fine-tuned CLIP model saved to {config.FINETUNED_MODEL_PATH}")

if __name__ == '__main__':

    main()