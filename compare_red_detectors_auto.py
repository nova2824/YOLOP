import cv2, os, numpy as np

ROOT = r"C:\Users\naray_he7wm7m\YOLOP"
video_path = os.path.join(ROOT, r"tools\inference\run_20251201_002433\source.mp4")
frames_dir = os.path.join(ROOT, "debug_frames")
out = os.path.join(ROOT, "debug_masks_compare")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(out, exist_ok=True)

# extract sample frames if not present
cap = cv2.VideoCapture(video_path)
frames_to_save = [0,1,2,10,50,100,500,1000]
for i in frames_to_save:
    dst = os.path.join(frames_dir, f"frame_{i:06d}.jpg")
    if os.path.exists(dst):
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, f = cap.read()
    if not ret:
        print("frame", i, "not found, skipping")
        continue
    cv2.imwrite(dst, f)
cap.release()
print("Sample frames ready in", frames_dir)

def hsv_red_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower1 = np.array((0,60,60), dtype=np.uint8)
    upper1 = np.array((12,255,255), dtype=np.uint8)
    lower2 = np.array((160,60,60), dtype=np.uint8)
    upper2 = np.array((180,255,255), dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1,m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def rgb_diff_mask(img):
    b,g,r = cv2.split(img.astype(int))
    mask = (r > 100) & ((r - g) > 40) & ((r - b) > 40)
    mask = (mask.astype('uint8') * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# run compare on saved sample frames
for fname in sorted(os.listdir(frames_dir)):
    if not fname.lower().endswith(('.jpg','.png')): continue
    img = cv2.imread(os.path.join(frames_dir, fname))
    hsvm = hsv_red_mask(img)
    rgbm = rgb_diff_mask(img)
    vis = cv2.hconcat([
        cv2.resize(img, (640,360)),
        cv2.cvtColor(cv2.resize(hsvm,(640,360)), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(cv2.resize(rgbm,(640,360)), cv2.COLOR_GRAY2BGR)
    ])
    outname = os.path.join(out, fname.replace('frame','compare'))
    cv2.imwrite(outname, vis)

print("Wrote comparisons to", out)
