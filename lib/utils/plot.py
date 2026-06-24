import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

def plot_img_and_mask(img, mask, index, epoch, save_dir):
    """
    Plot input image and corresponding segmentation mask(s) side by side and save the figure.
    """
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1, figsize=(15, 5))
    
    ax[0].set_title('Input image')
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i+1})')
            ax[i + 1].imshow(mask[:, :, i], cmap='gray')
    else:
        ax[1].set_title('Output mask')
        ax[1].imshow(mask, cmap='gray')
    
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/batch_{epoch}_{index}_seg.png")
    plt.close()


def show_seg_result(img, masks, palette, overlay_weight=0.5, is_demo=False):
    """
    Overlay segmentation masks on the original image.
    
    Args:
        img (np.array): Original image (H, W, 3) in BGR format.
        masks (tuple): Tuple of masks, e.g., (drivable_area_mask, lane_line_mask).
        palette (list): List of colors, each color is a tuple/list of 3 ints (B,G,R).
        overlay_weight (float): Weight for overlaying masks on image.
        is_demo (bool): Whether to show live OpenCV window.
    
    Returns:
        np.array: Image with overlay (BGR).
    """
    da_mask, ll_mask = masks

    # Combine masks by encoding classes as integers
    # For example: 0 = background, 1 = drivable area, 2 = lane lines
    color_mask = da_mask.astype(np.uint8) + (ll_mask.astype(np.uint8) * 2)
    
    # Resize mask to match image size
    color_mask = cv2.resize(color_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Prepare empty color segmentation image (H, W, 3)
    color_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # Apply palette colors where mask equals label
    for label, color in enumerate(palette):
        color_seg[color_mask == label] = color  # Assign color for each class label
    
    # Overlay the color mask on original image (BGR)
    overlay = cv2.addWeighted(img, 1 - overlay_weight, color_seg, overlay_weight, 0)

    if is_demo:
        cv2.imshow("YOLOP Segmentation Overlay", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
    
    return overlay


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    Draw one bounding box on the image.
    
    Args:
        x (list/tuple): Bounding box [x1, y1, x2, y2].
        img (np.array): Image to draw on (modified in place).
        color (list/tuple): Color for the box in BGR.
        label (str): Optional label text.
        line_thickness (int): Thickness of box lines.
    """
    tl = line_thickness or max(round(0.002 * (img.shape[0] + img.shape[1]) / 2), 1)
    color = color or [random.randint(0, 255) for _ in range(3)]

    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2_label = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(img, c1, c2_label, color, -1, cv2.LINE_AA)  # filled box for label background
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    print("This module is for visualization utilities and should be imported into your main script.")
