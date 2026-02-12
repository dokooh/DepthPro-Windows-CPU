# LICENSE_2 applies to this file
# Author JZ from LatteByte.ai 2024

import depth_pro
import matplotlib.pyplot as plt
import torch
import os
import cv2
import csv
import numpy as np

INPUT_IMAGE_PATH = "/kaggle/input/digsite-images/Construction - 12.jpg"


def _to_float(value, default=1.0):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().flatten()[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return float(value[0])
    return float(value)


def main():
    current_dir = os.path.dirname(__file__)
    debug_dir = os.path.join(current_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    image_path = INPUT_IMAGE_PATH
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Load image for both model inference and contour-based object detection.
    rgb_pil, _, f_px = depth_pro.load_rgb(image_path)
    rgb_np = np.array(rgb_pil)

    image_tensor = transform(rgb_pil)
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.as_tensor(image_tensor)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Run inference.
    with torch.no_grad():
        prediction = model.infer(image_tensor, f_px=f_px)

    # Depth output (meters)
    depth_np = prediction["depth"].squeeze().cpu().numpy().astype(np.float32)

    # Save raw depth array
    depth_npy_path = os.path.join(debug_dir, "depth_output.npy")
    np.save(depth_npy_path, depth_np)

    # Save depth visualization
    depth_vis_path = os.path.join(debug_dir, "depth_visualization.png")
    plt.figure(figsize=(10, 6))
    plt.imshow(depth_np, cmap="plasma")
    plt.colorbar(label="Depth (meters)")
    plt.title("Inferred Depth Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(depth_vis_path, dpi=200)
    plt.close()

    # Detect objects with simple contour extraction on the RGB image.
    gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    f_px_value = _to_float(f_px, default=1.0)
    area_threshold_px = 500
    object_rows = []
    overlay = rgb_np.copy()
    object_id = 0

    for contour in contours:
        area_px = cv2.contourArea(contour)
        if area_px < area_threshold_px:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        depth_roi = depth_np[y : y + h, x : x + w]
        valid_depth = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]
        if valid_depth.size == 0:
            continue

        object_id += 1
        z_med = float(np.median(valid_depth))
        width_m = (w * z_med) / max(f_px_value, 1e-6)
        height_m = (h * z_med) / max(f_px_value, 1e-6)
        area_m2 = width_m * height_m

        object_rows.append(
            {
                "object_id": object_id,
                "bbox_x": int(x),
                "bbox_y": int(y),
                "bbox_w_px": int(w),
                "bbox_h_px": int(h),
                "area_px": float(area_px),
                "median_depth_m": z_med,
                "width_m": float(width_m),
                "height_m": float(height_m),
                "area_m2": float(area_m2),
            }
        )

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"ID {object_id}: {width_m:.2f}m x {height_m:.2f}m",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Save object-size measurements to CSV
    object_csv_path = os.path.join(debug_dir, "object_sizes.csv")
    csv_headers = [
        "object_id",
        "bbox_x",
        "bbox_y",
        "bbox_w_px",
        "bbox_h_px",
        "area_px",
        "median_depth_m",
        "width_m",
        "height_m",
        "area_m2",
    ]
    with open(object_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(object_rows)

    # Save object overlay visualization
    overlay_path = os.path.join(debug_dir, "object_detection_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Input image: {image_path}")
    print(f"Saved depth array: {depth_npy_path}")
    print(f"Saved depth visualization: {depth_vis_path}")
    print(f"Saved object measurements: {object_csv_path}")
    print(f"Saved object overlay: {overlay_path}")
    print(f"Objects measured: {len(object_rows)}")


if __name__ == "__main__":
    main()
