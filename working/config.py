# src/config.py

# --- Foundational Paths ---
SYNTHETIC_DATA_PATH = "data/synthetic"
REAL_DATA_PATH = "data/real"
# Use a new, descriptive name for this experiment's processed data
PROCESSED_DATA_DIR = "working_processed_data_binary_s2r" 
ARTIFACTS_DIR = "working_artifacts_binary_s2r"

# --- Model Paths ---
FINETUNED_MODEL_PATH = f"{ARTIFACTS_DIR}/clip_finetuned_sim2real"

# --- CLIP and Model Configuration ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# --- Class Mappings and Text Prompts for the BINARY task ---
CLASS_MAPPING = {
    "empty_room": 0, 
    "1_person": 1,
}

TEXT_DESCRIPTORS = {
    0: "A radar heatmap of an empty room with only static background and sensor noise.",
    1: "A radar heatmap showing the distinct micro-doppler signature of a single person walking."
}

CLASS_NAMES = list(CLASS_MAPPING.keys())

# --- Hyperparameters ---
FINETUNE_EPOCHS = 50
FINETUNE_BATCH_SIZE = 32
FINETUNE_LR = 5e-6
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 0.01
EVAL_BATCH_SIZE = 64

