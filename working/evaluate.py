# src/evaluate.py

import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import os
from . import config, dataset, utils

def run_zero_shot_eval(model, processor, test_loader, device):

    # ... (This function does not need changes)
    model.to(device)
    model.eval()
    text_prompts = [config.TEXT_DESCRIPTORS[i] for i in range(len(config.CLASS_NAMES))]
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label']
            text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
            outputs = model(pixel_values=pixel_values, **text_inputs)
            logits_per_image = outputs.logits_per_image
            preds = logits_per_image.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    return torch.cat(all_preds), torch.cat(all_labels)

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # --- FIX: Point to the correct test set file ---
    test_h5_path = os.path.join(config.PROCESSED_DATA_DIR, 'test_frames.h5')
    # stats_path = os.path.join(config.PROCESSED_DATA_DIR, "norm_stats.json")
    stats_path = os.path.join(config.PROCESSED_DATA_DIR, "test_norm_stats.json")
    print(f"Using test data from: {test_h5_path} with stats at {stats_path}")
    if not os.path.exists(test_h5_path):
        raise FileNotFoundError(f"Test data not found at {test_h5_path}. Please run preprocess.py first.")

    class_names = config.CLASS_NAMES

    # In this supervised setup, we only evaluate our fine-tuned model.
    # The "baseline" evaluation is less meaningful here.
    print("\n--- Running Evaluation with Fine-tuned CLIP on the Test Set ---")
    try:
        tuned_model = CLIPModel.from_pretrained(config.FINETUNED_MODEL_PATH)
        tuned_processor = CLIPProcessor.from_pretrained(config.FINETUNED_MODEL_PATH)
        test_dataset = dataset.RadarFrameDataset(test_h5_path, stats_path, tuned_processor)
        test_loader = DataLoader(test_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
        tuned_preds, true_labels = run_zero_shot_eval(tuned_model, tuned_processor, test_loader, device)

        output_path_tuned = os.path.join(config.ARTIFACTS_DIR, "confusion_matrix_finetuned.png")
        utils.plot_advanced_confusion_matrices(
            true_labels.tolist(), tuned_preds.tolist(), class_names,
            output_path_tuned, title_prefix="Fine-tuned CLIP Evaluation"
        )
    except OSError:
        print(f"❌ Fine-tuned model not found at {config.FINETUNED_MODEL_PATH}. Please run finetune_clip.py first.")

if __name__ == '__main__':
    main()