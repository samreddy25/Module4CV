import cv2
import numpy as np
import argparse
from pathlib import Path

def to_grayscale(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

def gaussian_smooth(gray):
    return cv2.GaussianBlur(gray, (5,5), 1.5)

def detect_edges(smoothed):
    return cv2.Canny(smoothed, 50, 150)

def hot_region(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    heat = hsv[:,:,2]  # value channel carries thermal energy

    heat = cv2.GaussianBlur(heat,(5,5),1.5)

    t = np.percentile(heat,80)
    mask = np.zeros_like(heat,dtype=np.uint8)
    mask[heat>=t] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,k,iterations=1)

    return mask

def clean_region(mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def best_contour_mask(edges, region):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    edges = cv2.dilate(edges, k, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)==0:
        return np.zeros_like(edges)

    best_score = 0
    best_mask = None

    for c in cnts:
        temp = np.zeros_like(edges)
        cv2.drawContours(temp,[c],-1,255,-1)

        overlap = cv2.bitwise_and(temp,region)
        score = np.sum(overlap)

        if score > best_score:
            best_score = score
            best_mask = temp

    if best_mask is None:
        best_mask = np.zeros_like(edges)

    return best_mask

def draw_boundary(bgr, mask):
    overlay = bgr.copy()
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts)==0:
        return overlay

    cv2.drawContours(overlay,cnts,-1,(0,255,0),2)
    return overlay


def segment(image):
    gray = to_grayscale(image)
    smooth = gaussian_smooth(gray)
    edges = detect_edges(smooth)

    region = hot_region(image)
    region = clean_region(region)

    mask = best_contour_mask(edges, region)
    mask = solidify(mask, region)

    return gray, smooth, edges, region, mask

def solidify(mask, region):
    # ensure object interior remains filled using heat prior
    combined = cv2.bitwise_or(mask, region)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)

    # keep only component touching original mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)

    if n <= 1:
        return combined

    # find component overlapping contour mask
    best = 0
    best_overlap = 0

    for i in range(1,n):
        comp = np.zeros_like(mask)
        comp[labels==i] = 255
        overlap = np.sum(cv2.bitwise_and(comp, mask))
        if overlap > best_overlap:
            best_overlap = overlap
            best = i

    out = np.zeros_like(mask)
    out[labels==best] = 255
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_dir", default="output")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(in_path))
    if img is None:
        raise FileNotFoundError(in_path)

    gray, smooth, edges, region, mask = segment(img)
    overlay = draw_boundary(img, mask)

    cv2.imwrite(str(out_dir/"1_gray.png"), gray)
    cv2.imwrite(str(out_dir/"2_smooth.png"), smooth)
    cv2.imwrite(str(out_dir/"3_edges.png"), edges)
    cv2.imwrite(str(out_dir/"4_hotregion.png"), region)
    cv2.imwrite(str(out_dir/"5_mask.png"), mask)
    cv2.imwrite(str(out_dir/"6_overlay.png"), overlay)

    print("done")

if __name__ == "__main__":
    main()
