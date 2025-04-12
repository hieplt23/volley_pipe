import cv2

def load_mask(mask_path, frame_width=1920, frame_height=1080):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot load mask image: {mask_path}")

    mask_h, mask_w = mask.shape
    if mask_w != frame_width or mask_h != frame_height:
        print(f"Resizing mask from {mask_w}x{mask_h} to {frame_width}x{frame_height}")
        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

def foot_pos(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, y2