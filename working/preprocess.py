# src/preprocess.py
import os
import json
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob
from scipy.signal import windows
from scipy.ndimage import median_filter, gaussian_filter
from sklearn.model_selection import train_test_split
from . import config
import matplotlib.pyplot as plt
import imageio


# --- RDM Processing Configuration for REAL (JSON) Data ---
NUM_CHIRPS = 128
NUM_SAMPLES = 128
NUM_RX = 3
RANGE_FFT_SIZE = NUM_SAMPLES
DOPPLER_FFT_SIZE = NUM_CHIRPS * 2
EPS = 1e-6
range_win = windows.blackmanharris(NUM_SAMPLES)
doppler_win = windows.blackmanharris(NUM_CHIRPS)
RANGE_PAD = NUM_SAMPLES
DOPPLER_PAD = NUM_CHIRPS
RANGE_FFT_LEN = NUM_SAMPLES * 2
DOPPLER_FFT_LEN = NUM_CHIRPS * 2


# Bin axes for visualization
doppler_bins = np.arange(-NUM_CHIRPS, NUM_CHIRPS)
range_bins = np.arange(RANGE_FFT_SIZE)


def process_frame_from_json(frame_data, moving_avg_state, moving_avg_alpha=0.6, mti_alpha=1.0):
    """Applies the advanced RDM processing pipeline to a single raw JSON data frame."""
    final_rdm = np.zeros((RANGE_FFT_SIZE, DOPPLER_FFT_SIZE), dtype=np.complex64)
    updated_mavg = moving_avg_state.copy()

    for ant in range(NUM_RX):
        key = f"RX{ant+1}"
        raw_list = frame_data.get(key)
        if raw_list is None: return None, moving_avg_state
        
        flat = np.array(raw_list)
        if flat.size != NUM_CHIRPS * NUM_SAMPLES * 2: return None, moving_avg_state

        reshaped = flat.reshape((NUM_CHIRPS, NUM_SAMPLES, 2))
        raw = reshaped[...,0] + 1j * reshaped[...,1]

        raw -= raw.mean(axis=1, keepdims=True)
        fft_rng = np.fft.fft(raw * range_win, axis=1) / NUM_SAMPLES
        
        rng_half = fft_rng[:, :RANGE_FFT_SIZE] 
        
        fft1d = rng_half.T * doppler_win
        fft_dopp = np.fft.fft(fft1d, n=DOPPLER_FFT_SIZE, axis=1) / NUM_CHIRPS

        updated_mavg[ant] = fft_dopp * moving_avg_alpha + moving_avg_state[ant] * (1 - moving_avg_alpha)
        mti_out = fft_dopp - updated_mavg[ant] * mti_alpha
        
        dop_shift = np.fft.fftshift(mti_out, axes=1)
        final_rdm += dop_shift

    return final_rdm, updated_mavg

# def process_synthetic_csv_source(input_path: str):
#     """Finds and loads pre-calculated RDMs from CSV files."""
#     all_rdms, all_labels = [], []
#     for class_name, label_idx in config.CLASS_MAPPING.items():
#         class_dir = os.path.join(input_path, class_name)
#         if not os.path.isdir(class_dir): continue
        
#         files = sorted(glob(os.path.join(class_dir, "*.csv")))
#         for fp in tqdm(files, desc=f"Loading Synthetic CSVs ({class_name})"):
#             rdm = np.transpose(np.loadtxt(fp, delimiter=','), axes=(1, 0))

#             # --- START OF FIX ---
#             # The synthetic data is missing the crucial fftshift on the Doppler axis.
#             # We apply it here to match the real-data processing pipeline and ensure
#             # physical correctness (negative Doppler for 'towards' motion).
#             # The Doppler dimension is axis 0 (length NUM_CHIRPS).
#             rdm = np.fft.fftshift(rdm, axes=0)
#             # --- END OF FIX ---

#             if rdm.shape != (NUM_CHIRPS, NUM_SAMPLES):
#                 print(f"⚠️  {fp} has unexpected shape {rdm.shape}, skipping")
#                 continue
#             else:
#                 print(f"Loaded {fp} with shape {rdm.shape}")
#             # The synthetic data is clean and already processed into RDMs.
#             # We will convert it to a consistent dB scale.
            
#             all_rdms.append(rdm)
#             all_labels.append(label_idx)

#     if not all_rdms: return None, None
    
#     # print("Converting synthetic data to dB scale...")
#     # db_list = [20*np.log10(np.abs(r)+EPS) for r in all_rdms]
#     # return np.array(db_list), np.array(all_labels)
#     return np.array(all_rdms), np.array(all_labels)


def process_synthetic_csv_source(input_path: str):
    """Finds and loads pre-calculated RDMs from CSV files."""
    all_rdms, all_labels = [], []
    for class_name, label_idx in config.CLASS_MAPPING.items():
        class_dir = os.path.join(input_path, class_name)
        if not os.path.isdir(class_dir): continue
        
        files = sorted(glob(os.path.join(class_dir, "*.csv")))
        for fp in tqdm(files, desc=f"Loading Synthetic CSVs ({class_name})"):
            rdm = np.transpose(np.loadtxt(fp, delimiter=','), axes=(1, 0))

            # Step 1: Center the Doppler spectrum (still required).
            rdm = np.fft.fftshift(rdm, axes=0)
            
            # --- START OF FIX ---
            # The user observed the RDM is rotated 180 degrees from physical reality.
            # (Positive Doppler for 'towards', increasing range for 'towards').
            # We apply a 180-degree rotation to correct both axes simultaneously.
            rdm = np.rot90(rdm, k=2)
            # rdm = np.rot90(rdm, k=2) # This already includes the vertical flip

            # --- END OF FIX ---

            if rdm.shape != (NUM_CHIRPS, NUM_SAMPLES):
                print(f"⚠️  {fp} has unexpected shape {rdm.shape}, skipping")
                continue
            
            all_rdms.append(rdm)
            all_labels.append(label_idx)

    if not all_rdms: return None, None
    
    return np.array(all_rdms), np.array(all_labels)



def process_frame(frame_data, moving_avg_state, moving_avg_alpha, mti_alpha):
    """
    Applies preprocessing: DC removal, windowing, zero-padding, FFT,
    half-spectrum extract + energy compensation, Doppler FFT,
    moving-average + MTI filtering, and coherent sum.

    Returns:
        final_rdm: complex RDM [range_bins, doppler_bins*2]
        updated_mavg: updated moving-average state per antenna
    """
    final_rdm = np.zeros((RANGE_FFT_SIZE, DOPPLER_FFT_SIZE), dtype=np.complex64)
    updated_mavg = moving_avg_state.copy()

    for ant in range(NUM_RX):
        key = f"RX{ant+1}"
        raw_list = frame_data.get(key)
        if raw_list is None:
            return None, moving_avg_state

        flat = np.array(raw_list)
        if flat.size != NUM_CHIRPS * NUM_SAMPLES * 2:
            return None, moving_avg_state

        # Reconstruct complex IQ
        reshaped = flat.reshape((NUM_CHIRPS, NUM_SAMPLES, 2))
        raw = reshaped[...,0] + 1j * reshaped[...,1]

        # DC removal
        raw -= raw.mean(axis=1, keepdims=True)
        # Range window + zero-pad
        win_rng = raw * range_win
        zp_rng = np.pad(win_rng, ((0,0),(0,RANGE_PAD)), mode='constant')
        # Range FFT + energy compensation
        fft_rng = np.fft.fft(zp_rng, axis=1) / NUM_SAMPLES
        rng_half = 2 * fft_rng[:, :RANGE_FFT_SIZE]

        # Doppler prep: transpose, window, pad
        fft1d = rng_half.T * doppler_win
        # fft1d = rng_half * doppler_win
        zp_dopp = np.pad(fft1d, ((0,0),(0,DOPPLER_PAD)), mode='constant')
        # Doppler FFT
        fft_dopp = np.fft.fft(zp_dopp, axis=1) / NUM_CHIRPS

        # Moving-average update and MTI subtraction
        updated_mavg[ant] = fft_dopp * moving_avg_alpha + moving_avg_state[ant] * (1 - moving_avg_alpha)
        mti_out = fft_dopp - updated_mavg[ant] * mti_alpha

        # Center zero-Doppler
        dop_shift = np.fft.fftshift(mti_out, axes=1)

        # Sum antennas
        final_rdm += dop_shift

    return final_rdm, updated_mavg



# def create_rdm_gif(
#     rdm_list, output_path, fps, cmap,
#     median_size=0, gauss_sigma=0.0, morph_size=0
# ):
#     """
#     Converts complex RDM list to dB, applies spatial noise-reduction filters,
#     and writes GIF with specified colormap.

#     - median_size: kernel for median filter (0 = off)
#     - gauss_sigma: sigma for Gaussian smoothing (0 = off)
#     - morph_size: size for morphological opening (0 = off)
#     """
#     # Convert to dB
#     db_list = [20*np.log10(np.abs(r)+EPS) for r in rdm_list]

#     # Optional median smoothing
#     if median_size and median_size > 1:
#         db_list = [median_filter(f, size=(median_size, median_size)) for f in db_list]

#     # if mean_size and mean_size > 1:
#     #     db_list = [mean_size(f, size=(mean_size, mean_size)) for f in db_list]


#     # Optional Gaussian smoothing
#     if gauss_sigma and gauss_sigma > 0:
#         from scipy.ndimage import gaussian_filter
#         db_list = [gaussian_filter(f, sigma=gauss_sigma) for f in db_list]

#     # Optional morphological opening to remove small speckle
#     if morph_size and morph_size > 1:
#         from scipy.ndimage import grey_opening
#         db_list = [grey_opening(f, size=(morph_size, morph_size)) for f in db_list]


#     # Determine display range from raw (pre-filter) statistics
#     all_vals = np.concatenate([f.flatten() for f in db_list])
#     vmin = np.percentile(all_vals, 30)
#     vmax = np.percentile(all_vals, 99.9)
#     print(f"{cmap} scale: {vmin:.1f}–{vmax:.1f} dB")

#     # Generate frames
#     images = []
#     fig, ax = plt.subplots(figsize=(6,6))
#     for i, frame in enumerate(tqdm(db_list, desc=f"GIF frames ({cmap})")):
#         ax.clear()
#         ax.imshow(
#             frame,
#             origin='lower',
#             aspect='auto',
#             vmin=vmin,
#             vmax=vmax,
#             cmap=cmap,
#             extent=[doppler_bins[0], doppler_bins[-1], range_bins[0], range_bins[-1]]
#         )
#         ax.set_xlabel("Doppler Bin (neg= towards)")
#         ax.set_ylabel("Range Bin")
#         ax.set_title(f"RDM Frame {i+1}/{len(db_list)}")
#         fig.canvas.draw()
#         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         images.append(img)
#     plt.close(fig)

#     # Save GIF
#     imageio.mimsave(output_path, images, fps=fps, loop=0)
#     print(f"✅ Saved {output_path}")



def process_real_json_source(input_path: str):
    """Finds and processes raw radar data from JSON files."""
    files_to_process, labels_to_process = [], []
    for class_name, label_idx in config.CLASS_MAPPING.items():
        class_dir = os.path.join(input_path, class_name)
        if not os.path.isdir(class_dir): continue
        files = sorted(glob(os.path.join(class_dir, "*.json")))
        files_to_process.extend(files)
        labels_to_process.extend([label_idx] * len(files))
    
    if not files_to_process: return None, None

    mv_state = np.zeros((NUM_RX, RANGE_FFT_SIZE, DOPPLER_FFT_SIZE), dtype=complex)
    complex_rdms, valid_labels = [], []
    for i, fp in enumerate(tqdm(files_to_process, desc="Processing Real JSON")):
        try:
            with open(fp, 'r') as f:
                frame = json.load(f)
            # rdm, mv_state = process_frame_from_json(frame, mv_state)
            rdm, mv_state = process_frame(frame, mv_state, moving_avg_alpha=0.6, mti_alpha=1.0)
            if rdm is not None:
                complex_rdms.append(rdm)
                valid_labels.append(labels_to_process[i])
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    print("Applying filters to real data...")
    db_list = [20*np.log10(np.abs(r)+EPS) for r in complex_rdms]
    db_list = [median_filter(f, size=3) for f in db_list]
    db_list = [gaussian_filter(f, sigma=1) for f in db_list]

    all_vals = np.concatenate([f.flatten() for f in db_list])
    vmin = np.percentile(all_vals, 30)
    vmax = np.percentile(all_vals, 99.9)
    

    #TODO: REMOVE LATER
    # Generate frames
    # images = []
    # output_path="test.gif"
    # fig, ax = plt.subplots(figsize=(6,6))
    # for i, frame in enumerate(tqdm(db_list, desc=f"GIF frames ")):
    #     ax.clear()
    #     ax.imshow(
    #         frame,
    #         origin='lower',
    #         aspect='auto',
    #         vmin=vmin,
    #         vmax=vmax,
    #         cmap='viridis',
    #         extent=[doppler_bins[0], doppler_bins[-1], range_bins[0], range_bins[-1]]
    #     )
    #     ax.set_xlabel("Doppler Bin (neg= towards)")
    #     ax.set_ylabel("Range Bin")
    #     ax.set_title(f"RDM Frame {i+1}/{len(db_list)}")
    #     fig.canvas.draw()
    #     img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     images.append(img)
    # plt.close(fig)

    # # Save GIF
    # fps = 10
    # imageio.mimsave(output_path, images, fps=fps, loop=0)
    # print(f"✅ Saved {output_path}")


    # --- START OF FIX ---
    # The raw RDM has range_bin 0 (closest) at the top (row 0).
    # To match standard plotting conventions (y-axis increases upwards),
    # we must vertically flip each RDM. This makes the data physically intuitive.
    print("Applying vertical flip to real data for correct range orientation...")
    complex_rdms = [np.flipud(r) for r in complex_rdms]
    # --- END OF FIX ---

    print("Applying filters to real data...")
    db_list = [20*np.log10(np.abs(r)+EPS) for r in complex_rdms]
    db_list = [median_filter(f, size=3) for f in db_list]
    db_list = [gaussian_filter(f, sigma=1) for f in db_list]

    all_vals = np.concatenate([f.flatten() for f in db_list])
    vmin = np.percentile(all_vals, 30)
    vmax = np.percentile(all_vals, 99.9)
    
    return np.array(db_list), np.array(valid_labels), vmin, vmax
    # return np.array(db_list), np.array(valid_labels), vmin, vmax

def main():
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # --- Process Synthetic Data (from CSVs) for Training and Validation ---
    print(f"Processing synthetic data from CSV files {config.SYNTHETIC_DATA_PATH}...")

    synth_frames, synth_labels = process_synthetic_csv_source(config.SYNTHETIC_DATA_PATH)
    if synth_frames is None: 
        print("❌ FATAL ERROR: No synthetic CSV data found. Check SYNTHETIC_DATA_PATH in config.")
        return

    print("Splitting synthetic data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        synth_frames, synth_labels, test_size=0.2, stratify=synth_labels, random_state=42)

    # --- Process Real Data (from JSONs) for Testing ---
    real_frames, real_labels, real_vmin, real_vmax = process_real_json_source(config.REAL_DATA_PATH)
    if real_frames is None: 
        print("❌ FATAL ERROR: No real JSON data found. Check REAL_DATA_PATH in config.")
        return

    X_test, y_test = real_frames, real_labels
    
    # --- Save Normalization Stats (calculated from SYNTHETIC TRAIN set only) ---
    min_val = np.min(X_train)
    max_val = np.max(X_train)
    stats = {'min': float(min_val), 'max': float(max_val)}
    stats_path = os.path.join(config.PROCESSED_DATA_DIR, "norm_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\nNormalization stats saved to {stats_path}")

    # --- Save test set stats ---
    # all_vals = np.concatenate([f.flatten() for f in db_list])
    # vmin = np.percentile(all_vals, 30)
    # vmax = np.percentile(all_vals, 99.9)    
    stats = {'min': float(real_vmin), 'max': float(real_vmax)}
    stats_path = os.path.join(config.PROCESSED_DATA_DIR, "test_norm_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\nTest set normalization stats saved to {stats_path}")

    # --- Save all datasets to HDF5 files ---
    for name, (X, y) in zip(['train', 'val', 'test'], [(X_train, y_train), (X_val, y_val), (X_test, y_test)]):
        out_path = os.path.join(config.PROCESSED_DATA_DIR, f"{name}_frames.h5")
        with h5py.File(out_path, 'w') as hf:
            hf.create_dataset('data', data=X, compression="gzip")
            hf.create_dataset('labels', data=y)
        print(f"Saved {len(y)} frames to {out_path}")

if __name__ == '__main__':
    main()