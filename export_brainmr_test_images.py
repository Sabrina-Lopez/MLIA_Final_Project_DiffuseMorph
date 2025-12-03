import os
import argparse
import numpy as np
from pathlib import Path
from skimage import io


def load_npy(path: Path):
    arr = np.load(str(path))
    # Expect (N, 2, H, W)
    return arr


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def export_pairs(npy_path: Path, prefix: str, out_dir: Path):
    vol = load_npy(npy_path)
    # Shape should be (N, 2, H, W): [M, F]
    assert vol.ndim == 4 and vol.shape[1] == 2, f"Expected (N, 2, H, W), got {vol.shape}"
    n, _, h, w = vol.shape

    for i in range(n):
        mov = vol[i, 0]  # moving
        fix = vol[i, 1]  # fixed

        mov_img = normalize_to_uint8(mov)
        fix_img = normalize_to_uint8(fix)

        mov_name = f"{prefix}_{i+1}_M.png"
        fix_name = f"{prefix}_{i+1}_F.png"

        io.imsave(str(out_dir / mov_name), mov_img)
        io.imsave(str(out_dir / fix_name), fix_img)


def main():
    parser = argparse.ArgumentParser(
        description="Export brainMR test moving/fixed images from numpy file (N,2,H,W)."
    )
    parser.add_argument(
        "--image_npy",
        type=str,
        default="./datasets/brainMR/brain_test_image_final.npy",
        help="Path to brain_test_image_final.npy",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./datasets/brainMR/brain_test_images",
        help="Output directory for exported PNGs.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_npy)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting brainMR test image pairs from {image_path} to {out_dir}")
    export_pairs(image_path, prefix="brain_test", out_dir=out_dir)
    print("Done.")


if __name__ == "__main__":
    main()