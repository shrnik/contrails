"""
Train YOLOv11n-seg on GVCCS contrail segmentation dataset.

Converts GVCCS COCO-format annotations to YOLO segmentation format,
creates a dataset config, and launches training.

Usage:
    python train_yolo_seg_gvccs.py [--epochs 100] [--imgsz 1024] [--batch 8] [--device 0]
                                   [--contrail-types new old] [--convert-only] [--resume]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm


GVCCS_DIR = os.path.join(os.path.dirname(__file__), "GVCCS")
YOLO_DATASET_DIR = os.path.join(os.path.dirname(__file__), "GVCCS_yolo_seg")


def convert_coco_to_yolo_seg(split: str, contrail_types: list[str] | None = None):
    """
    Convert GVCCS COCO annotations to YOLO segmentation format.

    YOLO seg format per line: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
    All coordinates are normalized to [0, 1].

    Args:
        split: 'train' or 'test'
        contrail_types: List of contrail types to include ('new', 'old', or both).
                        Defaults to ['new', 'old'] (all contrails).
    """
    if contrail_types is None:
        contrail_types = ["new", "old"]

    annotations_path = os.path.join(GVCCS_DIR, split, "annotations.json")
    src_images_dir = os.path.join(GVCCS_DIR, split, "images")

    with open(annotations_path, "r") as f:
        coco = json.load(f)

    # Map split name for YOLO ('test' -> 'val' since YOLO expects train/val)
    yolo_split = "val" if split == "test" else split
    dst_images_dir = os.path.join(YOLO_DATASET_DIR, "images", yolo_split)
    dst_labels_dir = os.path.join(YOLO_DATASET_DIR, "labels", yolo_split)
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_labels_dir, exist_ok=True)

    # Build annotation lookup by image_id
    annotations_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann.get("type") not in contrail_types:
            continue
        img_id = ann["image_id"]
        annotations_by_image.setdefault(img_id, []).append(ann)

    images_by_id = {img["id"]: img for img in coco["images"]}

    n_images = 0
    n_annotations = 0
    n_skipped_empty_seg = 0

    for img_info in tqdm(coco["images"], desc=f"Converting {split}"):
        img_id = img_info["id"]
        fname = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        src_path = os.path.join(src_images_dir, fname)
        if not os.path.exists(src_path):
            continue

        # Symlink image to YOLO dataset dir (saves disk space)
        dst_img_path = os.path.join(dst_images_dir, fname)
        if not os.path.exists(dst_img_path):
            os.symlink(os.path.abspath(src_path), dst_img_path)

        # Write label file
        label_fname = Path(fname).stem + ".txt"
        label_path = os.path.join(dst_labels_dir, label_fname)

        anns = annotations_by_image.get(img_id, [])
        lines = []
        for ann in anns:
            for seg in ann["segmentation"]:
                pts = seg  # flat list [x1, y1, x2, y2, ...]
                if len(pts) < 6:  # need at least 3 points
                    n_skipped_empty_seg += 1
                    continue
                # Normalize coordinates
                normalized = []
                for i in range(0, len(pts), 2):
                    normalized.append(f"{pts[i] / w:.6f}")
                    normalized.append(f"{pts[i + 1] / h:.6f}")
                # class_id=0 (single class: contrail)
                lines.append("0 " + " ".join(normalized))
                n_annotations += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        n_images += 1

    print(f"  {split}: {n_images} images, {n_annotations} segmentation polygons")
    if n_skipped_empty_seg:
        print(f"  Skipped {n_skipped_empty_seg} polygons with < 3 points")


def create_dataset_yaml():
    """Create the YOLO dataset YAML config."""
    config = {
        "path": os.path.abspath(YOLO_DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "contrail",
        },
    }
    yaml_path = os.path.join(YOLO_DATASET_DIR, "dataset.yaml")
    os.makedirs(YOLO_DATASET_DIR, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Dataset config saved to {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11n-seg on GVCCS contrails")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="CUDA device (e.g. 0, 0,1, cpu, mps)")
    parser.add_argument(
        "--contrail-types",
        nargs="+",
        default=["new", "old"],
        choices=["new", "old"],
        help="Which contrail types to include (default: both)",
    )
    parser.add_argument("--convert-only", action="store_true", help="Only convert dataset, skip training")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--model", default="yolo26n-seg.pt", help="Base model (default: yolo11n-seg.pt)")
    parser.add_argument("--project", default="runs/gvccs_seg", help="Output project directory")
    parser.add_argument("--name", default="yolo11n_seg", help="Run name")
    args = parser.parse_args()

    # Step 1: Convert GVCCS to YOLO format
    if not args.resume:
        print("=" * 50)
        print("Step 1: Converting GVCCS to YOLO segmentation format")
        print(f"  Contrail types: {args.contrail_types}")
        print("=" * 50)

        # Clean previous conversion
        if os.path.exists(YOLO_DATASET_DIR):
            shutil.rmtree(YOLO_DATASET_DIR)

        convert_coco_to_yolo_seg("train", contrail_types=args.contrail_types)
        convert_coco_to_yolo_seg("test", contrail_types=args.contrail_types)

    yaml_path = create_dataset_yaml()

    if args.convert_only:
        print("Conversion complete. Skipping training.")
        return

    # Step 2: Train
    print("=" * 50)
    print("Step 2: Training YOLO segmentation model")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}, Image size: {args.imgsz}, Batch: {args.batch}")
    print(f"  Device: {args.device}")
    print("=" * 50)

    from ultralytics import YOLO

    if args.resume:
        # Resume from last checkpoint
        last_ckpt = os.path.join(args.project, args.name, "weights", "last.pt")
        if not os.path.exists(last_ckpt):
            raise FileNotFoundError(f"No checkpoint found at {last_ckpt}")
        model = YOLO(last_ckpt)
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            # Augmentation suited for sky/contrail images
            hsv_h=0.01,       # minimal hue shift (sky colors shouldn't change much)
            hsv_s=0.3,
            hsv_v=0.3,
            flipud=0.0,       # no vertical flip (contrails don't appear upside down in sky)
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,   # helpful for sparse instance segmentation
            # Training params
            patience=20,
            save=True,
            save_period=10,
            workers=8,
            exist_ok=True,
        )

    # Step 3: Validate
    print("=" * 50)
    print("Step 3: Validating best model")
    print("=" * 50)
    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    best_model = YOLO(best_path)
    metrics = best_model.val(data=yaml_path, imgsz=args.imgsz, device=args.device)
    print(f"\nValidation results:")
    print(f"  Box mAP50:    {metrics.box.map50:.4f}")
    print(f"  Box mAP50-95: {metrics.box.map:.4f}")
    print(f"  Mask mAP50:    {metrics.seg.map50:.4f}")
    print(f"  Mask mAP50-95: {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
