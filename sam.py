import cv2
import numpy as np
import argparse
from pathlib import Path
from transformers import Sam2Processor, Sam2Model
import torch
from PIL import Image

def centroid_from_image(gray):
    ys, xs = np.where(gray > np.percentile(gray, 85))
    if len(xs) == 0:
        h, w = gray.shape
        return w//2, h//2
    return int(np.mean(xs)), int(np.mean(ys))

def run_sam(image_bgr):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Sam2Model.from_pretrained("facebook/sam2-hiera-large").to(device)
    processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cx, cy = centroid_from_image(gray)

    input_points = [[[[cx, cy]]]]
    input_labels = [[[1]]]

    inputs = processor(
        images=pil_img,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"]
    )[0]

    best_idx = torch.argmax(outputs.iou_scores.squeeze()).item()
    mask = masks[0, best_idx].numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    return mask

def draw_overlay(bgr, mask):
    overlay = bgr.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0,255,0), 2)
    return overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path("sam_output")
    out_dir.mkdir(exist_ok=True)

    img = cv2.imread(str(in_path))
    if img is None:
        raise FileNotFoundError(in_path)

    mask = run_sam(img)
    overlay = draw_overlay(img, mask)

    cv2.imwrite(str(out_dir / f"{in_path.stem}_sam_mask.png"), mask)
    cv2.imwrite(str(out_dir / f"{in_path.stem}_sam_overlay.png"), overlay)

    print("Saved to sam_output")

if __name__ == "__main__":
    main()
