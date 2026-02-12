# LICENSE_2 applies to this file
# Author JZ from LatteByte.ai 2024

import depth_pro
import matplotlib.pyplot as plt
import torch
import os
import cv2
import csv
import numpy as np
import importlib

INPUT_IMAGE_PATH = "/kaggle/input/digsite-images/Construction - 12.jpg"
GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM_V3_MODEL_ID = "facebook/sam3-hiera-large"
SAM_FALLBACK_MODEL_ID = "facebook/sam2.1-hiera-large"
TEXT_PROMPT = "ladder. safety cones. digsite."


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


def _load_hf_models(device: torch.device):
    transformers = importlib.import_module("transformers")
    AutoProcessor = getattr(transformers, "AutoProcessor")
    AutoModelForZeroShotObjectDetection = getattr(
        transformers, "AutoModelForZeroShotObjectDetection"
    )
    SamProcessor = getattr(transformers, "SamProcessor")
    SamModel = getattr(transformers, "SamModel")

    grounding_processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_MODEL_ID
    ).to(device)
    grounding_model.eval()

    sam_model_id = SAM_V3_MODEL_ID
    try:
        sam_processor = SamProcessor.from_pretrained(sam_model_id)
        sam_model = SamModel.from_pretrained(sam_model_id).to(device)
    except Exception:
        sam_model_id = SAM_FALLBACK_MODEL_ID
        sam_processor = SamProcessor.from_pretrained(sam_model_id)
        sam_model = SamModel.from_pretrained(sam_model_id).to(device)
    sam_model.eval()

    return (
        grounding_processor,
        grounding_model,
        sam_processor,
        sam_model,
        sam_model_id,
    )


def _grounded_boxes(
    image_pil,
    text_prompt: str,
    grounding_processor,
    grounding_model,
    device: torch.device,
):
    inputs = grounding_processor(
        images=image_pil,
        text=text_prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = grounding_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=0.25,
        text_threshold=0.20,
        target_sizes=[image_pil.size[::-1]],
    )[0]

    boxes = results["boxes"].detach().cpu().numpy() if "boxes" in results else np.empty((0, 4), dtype=np.float32)
    labels = results.get("labels", [])
    scores = results["scores"].detach().cpu().numpy() if "scores" in results else np.array([], dtype=np.float32)
    return boxes, labels, scores


def _sam_masks_from_boxes(image_pil, boxes, sam_processor, sam_model, device: torch.device):
    if boxes.shape[0] == 0:
        return []

    input_boxes = [boxes.tolist()]
    sam_inputs = sam_processor(
        images=image_pil,
        input_boxes=input_boxes,
        return_tensors="pt",
    )
    sam_inputs = {k: v.to(device) for k, v in sam_inputs.items()}

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    post_masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.detach().cpu(),
        sam_inputs["original_sizes"].detach().cpu(),
        sam_inputs["reshaped_input_sizes"].detach().cpu(),
    )

    iou_scores = sam_outputs.iou_scores.detach().cpu().numpy()
    masks_tensor = post_masks[0]
    masks_np = masks_tensor.numpy() if hasattr(masks_tensor, "numpy") else np.array(masks_tensor)

    selected_masks = []
    for idx in range(boxes.shape[0]):
        if masks_np.ndim == 4:
            best_iou_idx = int(np.argmax(iou_scores[0, idx])) if iou_scores.ndim == 3 else int(np.argmax(iou_scores[idx]))
            mask = masks_np[idx, best_iou_idx] > 0
        elif masks_np.ndim == 3:
            mask = masks_np[idx] > 0
        else:
            continue
        selected_masks.append(mask)

    return selected_masks


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        grounding_processor,
        grounding_model,
        sam_processor,
        sam_model,
        sam_model_id,
    ) = _load_hf_models(device)

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

    # Detect objects with text prompts and segment with SAM loaded from Hugging Face Hub.
    boxes, labels, scores = _grounded_boxes(
        rgb_pil,
        TEXT_PROMPT,
        grounding_processor,
        grounding_model,
        device,
    )
    masks = _sam_masks_from_boxes(rgb_pil, boxes, sam_processor, sam_model, device)

    f_px_value = _to_float(f_px, default=1.0)
    area_threshold_px = 500
    object_rows = []
    overlay = rgb_np.copy()
    object_id = 0

    for idx, mask in enumerate(masks):
        if mask.shape != depth_np.shape:
            continue

        area_px = float(mask.sum())
        if area_px < area_threshold_px:
            continue

        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        w = x_max - x_min + 1
        h = y_max - y_min + 1

        valid_depth = depth_np[mask & np.isfinite(depth_np) & (depth_np > 0)]
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
                "label": str(labels[idx]) if idx < len(labels) else "unknown",
                "score": float(scores[idx]) if idx < scores.shape[0] else 0.0,
                "bbox_x": int(x_min),
                "bbox_y": int(y_min),
                "bbox_w_px": int(w),
                "bbox_h_px": int(h),
                "area_px": float(area_px),
                "median_depth_m": z_med,
                "width_m": float(width_m),
                "height_m": float(height_m),
                "area_m2": float(area_m2),
            }
        )

        color = np.array([0, 255, 0], dtype=np.uint8)
        overlay[mask] = (0.6 * overlay[mask] + 0.4 * color).astype(np.uint8)
        cv2.rectangle(overlay, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
        label_text = str(labels[idx]) if idx < len(labels) else "unknown"
        score_text = float(scores[idx]) if idx < scores.shape[0] else 0.0
        cv2.putText(
            overlay,
            f"ID {object_id} {label_text} ({score_text:.2f}): {width_m:.2f}m x {height_m:.2f}m",
            (x_min, max(20, y_min - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Save object-size measurements to CSV
    object_csv_path = os.path.join(debug_dir, "object_sizes_sam.csv")
    csv_headers = [
        "object_id",
        "label",
        "score",
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
    overlay_path = os.path.join(debug_dir, "sam_object_detection_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"Input image: {image_path}")
    print(f"Prompt: {TEXT_PROMPT}")
    print(f"Grounding model: {GROUNDING_MODEL_ID}")
    print(f"SAM model: {sam_model_id}")
    print(f"Saved depth array: {depth_npy_path}")
    print(f"Saved depth visualization: {depth_vis_path}")
    print(f"Saved object measurements: {object_csv_path}")
    print(f"Saved object overlay: {overlay_path}")
    print(f"Objects measured: {len(object_rows)}")


if __name__ == "__main__":
    main()
