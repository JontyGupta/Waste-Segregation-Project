"""
Prepare a downloaded YOLO dataset zip for training.

Workflow:
    1. Extract zip to a temp folder
    2. Auto-detect images/labels folders inside
    3. Remap class IDs using a mapping file
    4. Split into train/val
    5. Copy into data/images/ and data/labels/
"""

import os
import sys
import shutil
import random
import zipfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_zip(zip_path: str, extract_to: str) -> Path: 
    """Extract zip and return the extraction directory."""
    extract_dir = Path(extract_to)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"Extracted to {extract_dir}")
    return extract_dir


def find_yolo_folders(root: Path):
    """
    Auto-detect images/ and labels/folders inside extracted zip.
    Handles common structures:
      - root/images/, root/labels/
      - root/train/images/, root/train/labels/
      - root/dataset_name/images/, etc.
      - Returns dict: {"train": (images_dir, labels_dir), "val": (images_dir, labels_dir)}
    """
    results = {}

    #Pattern 1: root/train/images/root/train/labels/
    for split in ["train", "val", "test"]:
        img_dir = None
        lbl_dir = None
        for candidate in root.rglob(f"{split}"):
            if candidate.is_dir():
                # Check if images/ is inside train/
                if (candidate/ "images").is_dir():
                    img_dir = candidate / "images"
                    lbl_dir = candidate / "labels"
                # Check if train/ itself contains images
                elif any(f.suffix.lower() in IMG_EXTENSIONS for f in candidate.iterdir() if f.is_file()): 
                    img_dir = candidate
                    # Look for labels sibling
                    parent = candidate.parent 
                    if split == "train" and (parent /"labels" / "train").is_dir(): 
                        lbl_dir = parent / "labels" / "train"
                    elif split == "val" and (parent / "labels" / "val").is_dir():
                        lbl_dir = parent / "labels" / "val"

        if img_dir and lbl_dir and lbl_dir.is_dir():
            results[split] = (img_dir, lbl_dir)

    # Pattern 2: root/images/train/ + root/labels/train/
    if not results:
        for images_root in root.rglob("images"):
            if images_root.is_dir():
                labels_root = images_root.parent / "labels"
                if labels_root.is_dir():
                    for split in ["train", "val", "test"]:
                        img_split = images_root / split
                        lbl_split = labels_root / split
                        if img_split.is_dir() and lbl_split.is_dir(): 
                            results[split] = (img_split, lbl_split)

    # Pattern 3: flat root/images/ + root/labels/ (no splits)
    if not results:
        for images_dir in root.rglob("images"):
            if images_dir.is_dir():
                labels_dir = images_dir.parent / "labels"
                if labels_dir.is_dir():
                    results["all"] = (images_dir, labels_dir)
                    break
    
    return results


def read_yaml_from_zip(zip_path: str) -> dict:
    """
    Read the data.yaml / data.yml file from inside a zip.
    Returns parsed YAML as a dict.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for auto-mapping. Install via: pip install PyYAML")
        
    with zipfile.ZipFile(zip_path) as zf:
        yaml_files = [f for f in zf.namelist() if f.endswith(('.yaml', '.yml'))]
        if not yaml_files:
            return {}
        # Prefer data.yaml over others
        target = yaml_files[0]
        for yf in yaml_files:
            if os.path.basename(yf).startswith("data"):
                target = yf
                break
        return yaml.safe_load(zf.read(target).decode('utf-8'))
        
    
def read_target_classes(yaml_path: str = "data/yolo_dataset.yaml") -> Dict[int, str]:
    """
    Read your project's yolo_dataset.yaml and return (class_id: class_name).
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PYYAML is required for auto-mapping. Install via: pip install PyYAML")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})


def _normalize(name: str) -> str:
    """Normalize a class name for matching: lowercase, strip, replace separators."""
    return name.lower().strip().replace('-','_').replace(' ','_')


def _similarity(a: str, b: str) -> float:
    """Compute similarity score between two class names (8.0 to 1.8)."""
    na, nb = _normalize(a), _normalize(b)

    # Exact match
    if na == nb:
        return 1.0
    
    # One contains the other (e.g., "bottle" matches "plastic_bottle") 
    if na in nb or nb in na:
        return 0.85
    
    # Check if any word in one appears in the other 
    words_a = set(na.split('_')) 
    words_b = set(nb.split('_'))
    common = words_a & words_b
    if common:
        return 0.7 + 0.1 * len(common)
    
    # Fuzzy sequence match
    return SequenceMatcher (None, na, nb).ratio()


def auto_generate_mapping(
    zip_path: str,
    target_yaml: str = "data/yolo_dataset.yaml",
    min_score: float = 0.55,
    save_path: Optional[str] = None,
)-> Dict[int, int]:
    """
    Automatically generate a class ID mapping by fuzzy-matching class names 
    from the downloaded dataset's data.yaml against your yolo_dataset.yaml.

    Args:
        zip path: Path to the downloaded dataset zip.
        target_yaml: Path to your project's yolo_dataset.yaml.
        min_score: Minimum similarity score to accept a match (0.0-1.0).
        save path: If provided, save the generated mapping to this file.
    
    Returns:
        Dict mapping source_class_id -> target_class_id.
    """
    # Read source classes from zip
    src_data = read_yaml_from_zip(zip_path)
    src_names = src_data.get('names', {})
    if isinstance(src_names, list):
        src_names = {i: name for i, name in enumerate(src_names)}

    if not src_names:
        print("ERROR: No class names found in the dataset's YAML file.")
        return {}
    
    # Read target classes
    target_names = read_target_classes(target_yaml)
    if not target_names:
        print(f"ERROR: No class names found in {target_yaml}.")
        return {}
    
    print(f"\nSource dataset has {len(src_names)} classes")
    print(f"Your yolo dataset.yaml has {len(target_names)} classes")
    print(f"\nAuto-matching (min_score={min_score})...\n")
    
    mapping: Dict[int, int] = {}
    unmatched: List[Tuple[int, str]] = []

    for src_id, src_name in sorted(src_names.items()):
        best_score = 0.0
        best_target_id = -1
        best_target_name = ""
        
        for tgt_id, tgt_name in target_names.items():
            score = _similarity(src_name, tgt_name)
            if score > best_score:
                best_score = score
                best_target_id = tgt_id
                best_target_name = tgt_name

        if best_score >= min_score:
            mapping[src_id] = best_target_id
            status = "EXACT" if best_score == 1.0 else f"score={best_score:.2f}"
            print(" MATCHED: {src_id}: {src_name:<25} -> {best_target_id}: {best_target_name:<25} ({status})")
        else:
            unmatched.append((src_id, src_name))
            print(f" SKIPPED: {src_id}: {src_name:<25} -> (best was {best_target_name}, score={best_score:.2f})")
        
    print(f"\nResult: {len(mapping)} matched, {len(unmatched)} skipped")
    
    if unmatched:
        print("\nskipped classes (no good match found):")
        for sid, sname in unmatched:
            print(" {sid}: {sname}")
        
    #Save mapping to file if requested
    if save_path and mapping:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True) 
        with open(save_path, 'w') as f: 
            f.write("# Auto-generated class mapping\n") 
            f.write(f"# Source: {zip_path}\n") 
            f.write(f"a Target: {target_yaml}\n\n") 
            for src_id in sorted(mapping): 
                tgt_id = mapping[src_id] 
                src_n = src_names.get(src_id, '?') 
                tgt_n = target_names.get(tgt_id, '?')
                f.write(f"{src_id} {tgt_id}     # {src_n} -> {tgt_n}\n")
        print(f"\nMapping saved to: {save_path}") 

    return mapping


def read_mapping(mapping_file: str) -> Dict[int, int]:
    """
    Read class mapping file.
    Format per line: <source_id> <target_id>
    Lines starting with # are comments.
    """
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                mapping[int(parts[0])] = int(parts[1])
    return mapping


def remap_and_copy(
    src_images: Path,
    src_labels: Path,
    dst_images: Path,
    dst_labels: Path,
    class_mapping: Dict[int, int],
    prefix: str = "",
) -> dict:
    """Remap class IDs in label files and copy images + labels to destination."""
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
        
    stats = {"copied": 0, "skipped_no_label": 0, "skipped_empty": 0, "remapped_lines": 0, "dropped_lines": 0}

    for img_file in sorted(src_images.iterdir()):
        if img_file.suffix.lower() not in IMG_EXTENSIONS:
            continue
        
        label_file = src_labels / f"{img_file.stem}.txt"
        if not label_file.exists():
            stats["skipped_no_label"] += 1
            continue
        
        # Remap labels
        remapped = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                src_id = int(parts[0])
                if src_id in class_mapping: 
                    parts[0] = str(class_mapping[src_id]) 
                    remapped.append(" ".join(parts)) 
                    stats["remapped_lines"] += 1
                else:
                    stats["dropped_lines"] += 1

        if not remapped:
            stats["skipped_empty"] += 1
            continue

        new_name = f"{prefix}{img_file.stem}" if prefix else img_file.stem 
        shutil.copy2(img_file, dst_images/ f"{new_name}{img_file.suffix}") 
        with open(dst_labels / f"{new_name}.txt", "w") as f:
            f.write("\n".join(remapped) + "\n")
        stats["copied"] += 1
    
    return stats


def split_flat_data(
    images_dir: Path,
    labels_dir: Path,
    dst_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Split a flat images/labels folder into train/val."""
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS] 
    paired = [img for img in images if (labels_dir / f"{img.stem}.txt").exists()]
    
    random.seed(seed)
    random.shuffle(paired)
    split_idx = int(len(paired) * (1 - val_ratio))
    
    for img_list, split in [(paired[:split_idx], "train"), (paired[split_idx:], "val")]:
        img_dst = dst_root / "images" / split
        lbl_dst = dst_root / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img in img_list:
            shutil.copy2(img, img_dst / img.name)
            shutil.copy2(labels_dir / f"{img.stem}.txt", lbl_dst / f"{img.stem}.txt")

    print(f" Split: {split_idx} train / {len(paired) - split_idx} val")


def prepare(
    zip_path: str,
    mapping_file: str,
    output_dir: str = "data",
    prefix: str = "",
    val_ratio: float = 0.2,
    keep_existing: bool = True,
):
    """
    Full pipeline: extract zip → detect structure → remap → split → copy to data/.

    Args:
        zip_path: Path to downloaded dataset zip.
        mapping_file: Path to class mapping file.
        output_dir: Final output directory (data/).
        prefix: Filename prefix to avoid collisions between datasets.
        val_ratio: Fraction of data for validation (only used if dataset has no split).
        keep_existing: If True, adds to existing data. If False, clears data/ first.
    """
    output = Path(output_dir)
    temp_dir = Path("data/downloads/_temp_extract")
    
    # clean temp
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Step 1: Extract
    extract_dir = extract_zip(zip_path, str(temp_dir))

    # Step 2: Find structure
    folders = find_yolo_folders(extract_dir)
    if not folders:
        print("ERROR: Could not find images/ and labels/ folders in the zip.")
        print("Contents:")
        for item in sorted(extract_dir.rglob("*"))[:30]:
            print(f" {item.relative_to(extract_dir)}")
        shutil.rmtree(temp_dir)
        return
    
    print(f"Detected splits: {list(folders.keys())}")
    for split, (img, lbl) in folders.items():
        img_count = len([f for f in img.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
        lbl_count = len(list(lbl.glob("*.txt")))
        print(f" {split}: {img_count} images, {lbl_count} labels")
    
    # Step 3: Read mapping
    class_mapping = read_mapping(mapping_file) 
    print(f"Class mapping: {len(class_mapping)} source classes mapped")
    
    # Step 4: Remap and copy
    if not keep_existing:
        for split in ["train", "val"]:
            for sub in ["images", "labels"]:
                d = output / sub / split
                if d.exists():
                    shutil.rmtree(d)
    
    if "all" in folders:
        # No pre-split remap to temp, then split
        temp_imgs = temp_dir/"_remapped" / "images"
        temp_lbls = temp_dir/"_remapped" / "labels"
        src_imgs, src_lbls = folders["all"]
        stats = remap_and_copy(src_imgs, src_lbls, temp_imgs, temp_lbls, class_mapping, prefix)
        print(f"Remapped: {stats}")
        split_flat_data(temp_imgs, temp_lbls, output, val_ratio)
    else:
        # Already split
        for split in ["train", "val"]:
            if split not in folders:
                continue
            src_imgs, src_lbls = folders[split]
            dst_imgs = output/ "images" / split
            dst_lbls = output/ "labels" / split
            stats = remap_and_copy(src_imgs, src_lbls, dst_imgs, dst_lbls, class_mapping, prefix)
            print(f" {split}: {stats}")

    # Cleanup temp
    shutil.rmtree(temp_dir)

    # Final count
    for split in ["train", "val"]:
        img_dir = output/ "images" / split
        if img_dir.exists():
            count = len([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
            print("Final data/{split}: {count} images")
    
    print("\nDone! Ready to train with: python main.py--train-yolo")
   

def prepare_with_mapping(
    zip_path: str,
    class_mapping: Dict[int, int],
    output_dir: str = "data",
    prefix: str="",
    val_ratio: float = 0.2,
    keep_existing: bool = True,
):
    """
    Same as prepare() but accepts a mapping dict directly (used by --auto-map).
    """
    output = Path(output_dir)
    temp_dir = Path("data/downloads/_temp_extract")
        
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    extract_dir = extract_zip(zip_path, str(temp_dir))
        
    folders = find_yolo_folders(extract_dir)
    if not folders:
        print("ERROR: Could not find images/ and labels/ folders in the zip.")
        shutil.rmtree(temp_dir)
        return
    
    print(f"Detected splits: {list(folders.keys())}")
        
    if not keep_existing:
        for split in ["train", "val"]:
            for sub in ["images", "labels"]:
                d = output / sub / split
                if d.exists(): 
                    shutil.rmtree(d)
        
    if "all" in folders:
        temp_imgs = temp_dir /"_remapped" / "images"
        temp_lbls = temp_dir /"_remapped" / "labels"
        src_imgs, src_lbls = folders["all"]
        stats = remap_and_copy(src_imgs, src_lbls, temp_imgs, temp_lbls, class_mapping, prefix)
        print(f"Remapped: {stats}")
        split_flat_data(temp_imgs, temp_lbls, output, val_ratio)
    else:
        for split in ["train", "val"]:
            if split not in folders:
                continue
            src_imgs, src_lbls = folders[split]
            dst_imgs = output / "images" / split
            dst_lbls = output / "labels" / split
            stats = remap_and_copy(src_imgs, src_lbls, dst_imgs, dst_lbls, class_mapping, prefix)
            print(f" {split}: {stats}")
        
    shutil.rmtree(temp_dir)
        
    for split in ["train", "val"]:
        img_dir = output / "images" / split
        if img_dir.exists():
            count = len([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS])
            print(f"Final data/{split}: {count} images")

    print("\nDone! Ready to train with: python main.py --train-yolo")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare downloaded YOLO dataset for training") 
    parser.add_argument("--zip", required=True, help="Path to downloaded dataset .zip file")
    parser.add_argument("--mapping", default=None, help="Path to class mapping .txt file (skip if using --auto-map)")
    parser.add_argument("--auto-map", action="store_true", help="Auto-generate mapping by fuzzy-matching class names")
    parser.add_argument("--target-yaml", default="data/yolo_dataset.yaml", help="Your yolo_dataset.yaml path")
    parser.add_argument("--min-score", type=float, default=0.55, help="Min similarity score for auto sapping (0.0-1.0)")
    parser.add_argument("--output", default="data", help="Output directory (default: data)")
    parser.add_argument("--prefix", default="", help="Filename prefix (use when merging multiple dataset)")
    parser.add_argument ("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--fresh", action="store_true", help="Clear existing data before adding")
    args = parser.parse_args()

    if args.auto_map:
        # Auto-generate mapping and save it
        zip_stem = Path(args.zip).stem
        save_path = f"scripts/mappings/{zip_stem}_auto.txt"
        mapping = auto_generate_mapping(args.zip, args.target_yaml, args.min_score, save_path)
        if not mapping:
            print("No classes watched. check your dataset or lower min-score.")
            sys.exit(1)
        # use the auto-generated mapping directly
        prepare_with_mapping(args.zip, mapping, args.output, args.prefix, args.val_ratio, keep_existing=not args.fresh)
    elif args.mapping:
        prepare(args.zip, args.mapping, args.output, args.prefix, args.val_ratio, keep_existing=not args.fresh)
    else:
        print("ERROR: Provide either --mapping <file> or --auto-map")
        sys.exit(1)