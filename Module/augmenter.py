# -*- coding: utf-8 -*-
"""
이미지 + 라벨맵 동시 증강 (회전, 쉬프트, 플립 등)

INPUT:
  /home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/images       (원본 이미지, .jpg)
  /home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/labelmaps    (RGB labelmap, .png)

OUTPUT:
  /home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/aug_images
  /home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/aug_labelmaps

증강 종류:
- 좌우 뒤집기
- 상하 뒤집기
- 90°, 180°, 270° 회전
- 랜덤 회전 (-30 ~ +30도)
- 랜덤 쉬프트(가로/세로 각 10% 범위)
"""

import os
from glob import glob
import random
from PIL import Image, ImageOps, ImageChops

# ---------------- 경로 (이 파일 위치 기준) ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Train_DeepLab3/
DATA_DIR = os.path.join(BASE_DIR, "Data")

# 증강 대상 소스들 (이미지 폴더, 마스크 폴더). 추가 데이터셋은 여기에 한 줄씩 추가.
SOURCES = [
    (os.path.join(DATA_DIR, "Source1_Image"),
     os.path.join(DATA_DIR, "Source1_Labeling", "SegmentationClass")),
    (os.path.join(DATA_DIR, "Source2_Image"),
     os.path.join(DATA_DIR, "Source2_Labeling", "SegmentationClass")),
]
OUT_IMG_DIR = os.path.join(DATA_DIR, "Aug_Images")
OUT_MASK_DIR = os.path.join(DATA_DIR, "Aug_SegmentationClass")
# -----------------------------------------------------------


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


ensure_dir(OUT_IMG_DIR)
ensure_dir(OUT_MASK_DIR)


def random_shift(img, mask, max_shift_ratio=0.1):
    """이미지와 마스크를 동일하게 랜덤 쉬프트"""
    w, h = img.size
    max_dx = int(w * max_shift_ratio)
    max_dy = int(h * max_shift_ratio)

    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    img_shifted = ImageChops.offset(img, dx, dy)
    mask_shifted = ImageChops.offset(mask, dx, dy)

    return img_shifted, mask_shifted


def random_rotation(img, mask, max_angle=30):
    """랜덤 회전 (-max_angle ~ +max_angle)"""
    angle = random.uniform(-max_angle, max_angle)

    img_rot = img.rotate(angle, resample=Image.BILINEAR)
    mask_rot = mask.rotate(angle, resample=Image.NEAREST)

    return img_rot, mask_rot


def save_pair(stem, idx, img, mask):
    """저장 함수"""
    img.save(os.path.join(OUT_IMG_DIR, f"{stem}_aug_{idx:03d}.jpg"))
    mask.save(os.path.join(OUT_MASK_DIR, f"{stem}_aug_{idx:03d}.png"))


NUM_VARIANTS = 8   # augment() 1장당 생성하는 증강본 수 (idx 0~7)


def augment(stem, img_path, mask_path):
    # 이미 완전히 증강된 경우만 건너뜀. 부분 증강본은 지우고 다시 생성.
    existing = glob(os.path.join(OUT_IMG_DIR, f"{stem}_aug_*.jpg"))

    if len(existing) >= NUM_VARIANTS:
        print(f"[SKIP] {stem} 이미 증강됨 → 건너뜀")
        return

    if existing:
        print(f"[REDO] {stem} 부분 증강({len(existing)}개) → 정리 후 재생성")
        for f in existing:
            os.remove(f)
        for f in glob(os.path.join(OUT_MASK_DIR, f"{stem}_aug_*.png")):
            os.remove(f)

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    idx = 0

    # 원본도 학습 데이터에 포함
    save_pair(stem, idx, img, mask); idx += 1

    # 좌우 플립
    img2 = ImageOps.mirror(img)
    mask2 = ImageOps.mirror(mask)
    save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 회전
    # img2, mask2 = random_rotation(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 쉬프트
    # img2, mask2 = random_shift(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # 상하 플립
    img2 = ImageOps.flip(img)
    mask2 = ImageOps.flip(mask)
    save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 회전
    # img2, mask2 = random_rotation(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 쉬프트
    # img2, mask2 = random_shift(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # 90도 회전
    img2 = img.rotate(90, expand=True)
    mask2 = mask.rotate(90, expand=True, resample=Image.NEAREST)
    save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 회전
    # img2, mask2 = random_rotation(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 쉬프트
    # img2, mask2 = random_shift(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # 180도 회전
    img2 = img.rotate(180, expand=True)
    mask2 = mask.rotate(180, expand=True, resample=Image.NEAREST)
    save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 회전
    # img2, mask2 = random_rotation(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 쉬프트
    # img2, mask2 = random_shift(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # 270도 회전
    img2 = img.rotate(270, expand=True)
    mask2 = mask.rotate(270, expand=True, resample=Image.NEAREST)
    save_pair(stem, idx, img2, mask2); idx += 1

    # 랜덤 회전
    img2, mask2 = random_rotation(img, mask)
    save_pair(stem, idx, img2, mask2); idx += 1

    # 랜덤 쉬프트
    img2, mask2 = random_shift(img, mask)
    save_pair(stem, idx, img2, mask2); idx += 1

    print(f"[OK] Augmented {stem}")


def main():
    exts = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG"]
    for img_dir, mask_dir in SOURCES:
        if not os.path.isdir(img_dir):
            print(f"[WARN] 소스 폴더 없음: {img_dir} → 건너뜀")
            continue
        print(f"[SOURCE] {img_dir}")

        image_files = []
        for ext in exts:
            image_files.extend(glob(os.path.join(img_dir, ext)))

        for img_path in image_files:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, stem + ".png")

            if not os.path.isfile(mask_path):
                print(f"[WARN] Missing mask for {stem}, skipped")
                continue

            augment(stem, img_path, mask_path)

    print("Augmentation complete!")


if __name__ == "__main__":
    main()
