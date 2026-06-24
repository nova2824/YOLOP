#!/usr/bin/env python3
"""
evaluate_lane_video.py (FIXED VERSION)

Usage examples:
    # Ground-truth masks folder (binary PNG per frame, same filenames as frames in video)
    python evaluate_lane_video.py --pred_video pred.mp4 --gt_masks_folder gt_masks/ --out_dir eval_out

    # Ground-truth polylines (CSV with frame_index,x1,y1,x2,y2,...)
    python evaluate_lane_video.py --pred_video pred.mp4 --gt_polylines gt_polylines.csv --out_dir eval_out

    # No ground-truth: do self-consistency metrics only
    python evaluate_lane_video.py --pred_video pred.mp4 --out_dir eval_out

Notes:
 - Predicted video is assumed to contain the visualization output of your model (lane pixels highlighted).
 - If your pred video overlays lanes on original frames, the script tries to extract a binary mask by color-thresholding.
 - For best results with GT masks, ensure filenames or frame indices align.
"""
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib_cache"))
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import argparse
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage import img_as_bool
import json

# ---------- Utilities ----------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def extract_binary_mask_from_frame(frame, method='color', color_ranges=None):
    """
    Extract lane mask from YOLOP overlay video.
    Returns binary mask (uint8: 0 or 255).
    Supports:
        - 'color': HSV + RGB-difference hybrid (best for red lane overlays)
        - 'edge': fallback Canny-based mask
    """

    if method == 'color':
        # ----------------------------
        # 1) HSV-based RED + GREEN mask
        # ----------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Wide + tolerant HSV ranges for your night footage
        hsv_ranges = color_ranges or [
            ((0, 50, 50),   (12, 255, 255)),    # red low
            ((160, 50, 50), (180, 255, 255)),   # red high
            ((35, 40, 40),  (85, 255, 255)),    # green drivable area
        ]

        hsv_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lo, hi in hsv_ranges:
            lo = np.array(lo, dtype=np.uint8)
            hi = np.array(hi, dtype=np.uint8)
            hsv_mask |= cv2.inRange(hsv, lo, hi)

        # ----------------------------
        # 2) If HSV is too weak → use RGB-difference (strong for red lanes)
        # ----------------------------
        if hsv_mask.sum() < 500:   # Threshold tuned for your frames
            b, g, r = cv2.split(frame.astype(np.int16))
            rgb_mask = (
                (r > 120) &                # red must be bright enough
                ((r - g) > 40) &           # red dominates green
                ((r - b) > 40)             # red dominates blue
            ).astype(np.uint8) * 255
            mask = rgb_mask
        else:
            mask = hsv_mask

        # ----------------------------
        # 3) Morphological cleaning
        # ----------------------------
        try:
            # Remove tiny noise + close gaps
            mask_bool = img_as_bool(mask)
            mask_clean = remove_small_objects(mask_bool, min_size=60)
            mask_clean = binary_closing(mask_clean, disk(5))
            mask = (mask_clean.astype(np.uint8) * 255)
        except Exception:
            # Fallback if skimage not available
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    # --------------------------------------------------
    # Method = 'edge' (fallback)
    # --------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return (closed > 0).astype(np.uint8) * 255


def read_gt_mask_for_frame(gt_folder, frame_idx):
    """
    Read ground-truth mask for a frame. Handles multiple naming conventions.
    Returns uint8 binary mask or None if not found.
    """
    # search common extensions
    for ext in ('png','jpg','jpeg','bmp'):
        path = os.path.join(gt_folder, f"{frame_idx}.{ext}")
        if os.path.exists(path):
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None:  # BUG FIX #5: Handle corrupted files
                return None
            return (m > 127).astype(np.uint8)
    # try zero-padded names
    for ext in ('png','jpg'):
        for pad in (6,5,4,3,2):
            name = str(frame_idx).zfill(pad)
            path = os.path.join(gt_folder, f"{name}.{ext}")
            if os.path.exists(path):
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if m is None:  # BUG FIX #5: Handle corrupted files
                    return None
                return (m > 127).astype(np.uint8)
    return None

def iou(pred, gt):
    """Compute Intersection over Union between pred and gt masks."""
    predb = pred.astype(bool)
    gtb = gt.astype(bool)
    inter = np.logical_and(predb, gtb).sum()
    union = np.logical_or(predb, gtb).sum()
    if union > 0:
        return float(inter) / union
    else:
        # Both empty: perfect match
        return 1.0 if inter == 0 else 0.0

def precision_recall_f1(pred, gt):
    """Compute precision, recall, and F1 score."""
    predb = pred.astype(bool)
    gtb = gt.astype(bool)
    tp = np.logical_and(predb, gtb).sum()
    fp = np.logical_and(predb, np.logical_not(gtb)).sum()
    fn = np.logical_and(np.logical_not(predb), gtb).sum()
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def safe_json_serialize(obj):
    """Handle NaN and inf values in JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# ---------- Main evaluation ----------
def evaluate(pred_video, out_dir,
             gt_masks_folder=None, gt_polylines_csv=None,
             overlay_out=None, color_thresh_method='color',
             downsample=1):
    """
    Evaluate predicted lane video against ground truth (if provided).
    
    Args:
        pred_video: Path to prediction video
        out_dir: Output directory for results
        gt_masks_folder: Optional folder with GT binary masks
        gt_polylines_csv: Optional CSV with GT polylines (placeholder)
        overlay_out: Optional output path for overlay video
        color_thresh_method: 'color' or 'edge' for mask extraction
        downsample: Downsampling factor for speed
    
    Returns:
        Dictionary with results, dataframe, and summary metrics
    """
    # BUG FIX #9: Validate input file exists
    if not os.path.exists(pred_video):
        raise FileNotFoundError(f"Prediction video not found: {pred_video}")
    
    ensure_dir(out_dir)
    per_frame_results = []
    prev_mask = None
    processed = 0
    overlay_writer = None

    cap = cv2.VideoCapture(pred_video)
    
    # BUG FIX #6: Default FPS to 30 if not available
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0:
        fps_in = 30.0
        print(f"Warning: FPS not detected, using default {fps_in}")
    
    # BUG FIX #13: Proper division for width/height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / downsample)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / downsample)

    if overlay_out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        overlay_writer = cv2.VideoWriter(overlay_out, fourcc, max(1, fps_in), (width, height))
        # BUG FIX #12: Validate VideoWriter was created
        if not overlay_writer.isOpened():
            raise RuntimeError(f"Failed to create VideoWriter for {overlay_out}. Check codec availability.")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if downsample != 1:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        
        # Extract predicted mask
        pred_mask = extract_binary_mask_from_frame(frame, method=color_thresh_method)
        
        # BUG FIX #4: Morphological closing with proper type handling
        pred_mask = (binary_closing(pred_mask.astype(bool), disk(3)) * 255).astype(np.uint8)

        row = OrderedDict()
        row['frame'] = idx
        pred_pixel_count = int(np.count_nonzero(pred_mask))
        total_pixels = int(pred_mask.shape[0] * pred_mask.shape[1])
        row['pred_pixels'] = pred_pixel_count
        row['video_fps'] = fps_in

        # Compute temporal stability metrics
        if prev_mask is not None:
            t_iou = iou(pred_mask, prev_mask)
            row['temporal_iou'] = t_iou
            try:
                s = ssim(prev_mask, pred_mask, data_range=255)
            except Exception:
                s = float('nan')
            row['temporal_ssim'] = s
        else:
            row['temporal_iou'] = float('nan')
            row['temporal_ssim'] = float('nan')

        # Ground-truth metrics (if provided)
        if gt_masks_folder:
            gt = read_gt_mask_for_frame(gt_masks_folder, idx)
            if gt is not None:
                # BUG FIX #7: Proper shape comparison
                if gt.shape[:2] != pred_mask.shape[:2]:
                    gt = cv2.resize(gt.astype(np.uint8), (pred_mask.shape[1], pred_mask.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
                    gt = (gt > 0).astype(np.uint8)
                row['gt_pixels'] = int(gt.sum())
                row['iou'] = iou(pred_mask, gt)
                prec, rec, f1 = precision_recall_f1(pred_mask, gt)
                row['precision'] = prec
                row['recall'] = rec
                row['f1'] = f1
            else:
                row['gt_pixels'] = float('nan')
                row['iou'] = float('nan')
                row['precision'] = float('nan')
                row['recall'] = float('nan')
                row['f1'] = float('nan')
        else:
            row['gt_pixels'] = float('nan')
            row['iou'] = float('nan')
            row['precision'] = float('nan')
            row['recall'] = float('nan')
            row['f1'] = float('nan')

        # Detection ratio
        row['detection_ratio'] = float(pred_pixel_count) / total_pixels if total_pixels else 0.0

        per_frame_results.append(row)
        prev_mask = pred_mask.copy()
        idx += 1
        processed += 1

        # Overlay writing
        if overlay_writer is not None:
            overlay = frame.copy()
            mask_color = (0, 0, 255)
            # BUG FIX #15: Efficient blending only on masked pixels
            mask_bool = pred_mask.astype(bool)
            overlay[mask_bool] = cv2.addWeighted(
                overlay[mask_bool], 0.4,
                np.full_like(overlay[mask_bool], mask_color), 0.6, 0
            )
            overlay_writer.write(overlay)

    cap.release()
    if overlay_writer is not None:
        overlay_writer.release()

    if processed == 0:
        raise RuntimeError("No frames processed. Check pred_video path.")

    df = pd.DataFrame(per_frame_results)
    
    # Compute aggregates
    summary = {
        'num_frames': processed,
        'video_fps': float(df['video_fps'].dropna().median()),
        'avg_pred_pixels': float(df['pred_pixels'].mean()),
        'median_detection_ratio': float(df['detection_ratio'].median())
    }
    
    # BUG FIX #10: Proper NaN checking
    if not df['iou'].isna().all():
        summary['mean_iou'] = float(df['iou'].dropna().mean())
        summary['mean_f1'] = float(df['f1'].dropna().mean())
        summary['mean_precision'] = float(df['precision'].dropna().mean())
        summary['mean_recall'] = float(df['recall'].dropna().mean())

    # Save CSV
    csv_path = os.path.join(out_dir, "per_frame_metrics.csv")
    df.to_csv(csv_path, index=False)

    # BUG FIX #11: Safe JSON serialization with NaN handling
    summary_path = os.path.join(out_dir, "summary_metrics.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=safe_json_serialize)

    # Plots
    plt.figure(figsize=(10, 4))
    plt.plot(df['frame'], df['detection_ratio'], label='detection_ratio')
    plt.xlabel('frame')
    plt.ylabel('detection_ratio')
    plt.title('Detection ratio per frame')
    plt.grid(True)
    plt.tight_layout()
    det_plot = os.path.join(out_dir, 'detection_ratio.png')
    plt.savefig(det_plot)
    plt.close()

    iou_plot = None
    if not df['iou'].isna().all():
        plt.figure(figsize=(10, 4))
        plt.plot(df['frame'], df['iou'], label='IoU', alpha=0.8)
        plt.xlabel('frame')
        plt.ylabel('IoU')
        plt.title('IoU (pred vs GT) per frame')
        plt.grid(True)
        plt.tight_layout()
        iou_plot = os.path.join(out_dir, 'iou_per_frame.png')
        plt.savefig(iou_plot)
        plt.close()

    f1_plot = None
    if not df['f1'].isna().all():
        plt.figure(figsize=(10, 4))
        plt.plot(df['frame'], df['f1'], label='F1', alpha=0.8)
        plt.xlabel('frame')
        plt.ylabel('F1')
        plt.title('F1-score (pred vs GT) per frame')
        plt.grid(True)
        plt.tight_layout()
        f1_plot = os.path.join(out_dir, 'f1_per_frame.png')
        plt.savefig(f1_plot)
        plt.close()

    temp_plot = None
    if not df['temporal_iou'].isna().all():
        plt.figure(figsize=(10, 4))
        plt.plot(df['frame'], df['temporal_iou'], label='temporal_iou', alpha=0.8)
        plt.xlabel('frame')
        plt.ylabel('temporal_iou')
        plt.title('Temporal IoU between consecutive frames')
        plt.grid(True)
        plt.tight_layout()
        temp_plot = os.path.join(out_dir, 'temporal_iou.png')
        plt.savefig(temp_plot)
        plt.close()

    # Summary table
    table_path = os.path.join(out_dir, 'summary_table.csv')
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(table_path, index=False)

    results = {
        'per_frame_csv': csv_path,
        'summary_json': summary_path,
        'summary_table_csv': table_path,
        'plots': {
            'detection_ratio': det_plot,
            'iou': iou_plot,
            'f1': f1_plot,
            'temporal_iou': temp_plot
        },
        'overlay_video': overlay_out if overlay_out and os.path.exists(overlay_out) else None,
        'dataframe': df,
        'summary': summary
    }
    return results

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate lane-detection predicted video.")
    p.add_argument('--pred_video', required=True, help='Path to predicted output video')
    p.add_argument('--gt_masks_folder', default=None, 
                   help='Folder with ground-truth binary masks named by frame index (e.g., 0.png, 1.png)')
    p.add_argument('--gt_polylines', default=None, 
                   help='CSV of ground-truth polylines (placeholder).')
    p.add_argument('--out_dir', default='eval_out', 
                   help='Output folder for CSVs, plots, overlays')
    p.add_argument('--overlay_out', default=None, 
                   help='If set, write an overlay video for visual check (mp4)')
    p.add_argument('--color_method', default='color', choices=['color', 'edge'], 
                   help='Method to extract predicted mask from frames')
    p.add_argument('--downsample', type=int, default=1, 
                   help='Downsample factor for speed / smaller outputs')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = evaluate(args.pred_video,
                   args.out_dir,
                   gt_masks_folder=args.gt_masks_folder,
                   gt_polylines_csv=args.gt_polylines,
                   overlay_out=args.overlay_out,
                   color_thresh_method=args.color_method,
                   downsample=args.downsample)
    print("Evaluation finished.")
    print("Summary:")
    for k, v in res['summary'].items():
        print(f"  {k}: {v}")
    print("\nSaved artifacts:")
    print(" - Per-frame csv:", res['per_frame_csv'])
    print(" - Summary json:", res['summary_json'])
    print(" - Summary table CSV:", res['summary_table_csv'])
    for name, p in res['plots'].items():
        if p:
            print(f" - Plot ({name}): {p}")
    if res['overlay_video']:
        print(" - Overlay video:", res['overlay_video'])
    print("\nPer-frame preview (first 5 rows):")
    print(res['dataframe'].head().to_string(index=False))
