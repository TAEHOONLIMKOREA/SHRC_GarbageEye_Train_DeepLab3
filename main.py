import os
import re
from glob import glob
from collections import defaultdict

from Module.TrainMethods import data_generator, continue_train


# ── 경로 설정 (이 파일 위치 기준) ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # Train_DeepLab3/
DATA_DIR = os.path.join(BASE_DIR, "Data")

# augmenter.py 산출물 (원본 + 증강본)
AUG_IMG_DIR = os.path.join(DATA_DIR, "Aug_Images")
AUG_MASK_DIR = os.path.join(DATA_DIR, "Aug_SegmentationClass")

# 레이블맵 (현재 서빙 모델과 동일: 6클래스, 동일 순서/컬러)
LABEL_PATH = os.path.join(DATA_DIR, "Source1_Labeling", "labelmap.txt")

# 추가 학습(fine-tuning) 대상 = 현재 서빙 중인 모델
# 학습 후 새 모델은 같은 폴더(../Modules/GarbageEye/models/)에 저장됩니다.
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "Modules", "GarbageEye", "models",
    "continued_model_2026-06-25_18-13-55.keras"
)

# ── 학습 설정 ─────────────────────────────────────────────────
VAL_RATIO = 0.1          # 검증셋 비율
EPOCHS = 60              # 최대 epoch (EarlyStopping이 최적점에서 자동 중단)
LEARNING_RATE = 1e-5
PATIENCE = 5             # val_loss가 PATIENCE epoch 동안 개선 없으면 중단


def original_stem(path):
    """증강 파일명에서 원본 stem 추출 (GarbageSample_1_aug_003 → GarbageSample_1)"""
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"_aug_\d+$", "", base)


def main():
    # [0] 증강 데이터 존재 확인
    if not os.path.isdir(AUG_IMG_DIR) or len(os.listdir(AUG_IMG_DIR)) == 0:
        raise SystemExit(
            "증강 데이터가 없습니다. 먼저 아래 명령으로 증강을 수행하세요:\n"
            "    python Module/augmenter.py"
        )

    # [1] 이미지 ↔ 마스크 페어링 (파일명 stem 기준)
    images = sorted(glob(os.path.join(AUG_IMG_DIR, "*")))
    masks = sorted(glob(os.path.join(AUG_MASK_DIR, "*")))
    mask_by_key = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}

    pairs = []
    for im in images:
        key = os.path.splitext(os.path.basename(im))[0]
        if key in mask_by_key:
            pairs.append((im, mask_by_key[key]))
    if not pairs:
        raise SystemExit("이미지와 매칭되는 마스크를 찾지 못했습니다. 경로를 확인하세요.")

    # [2] train/val 분리 — 같은 원본의 증강본은 한쪽에만 (데이터 누수 방지)
    groups = defaultdict(list)
    for im, m in pairs:
        groups[original_stem(im)].append((im, m))

    stems = sorted(groups)
    step = max(1, int(round(1 / VAL_RATIO)))      # VAL_RATIO=0.1 → 10개마다 1개를 val로
    val_stems = set(stems[::step])

    train_pairs = [p for s in stems if s not in val_stems for p in groups[s]]
    val_pairs = [p for s in stems if s in val_stems for p in groups[s]]

    train_images, train_masks = zip(*train_pairs)
    val_images, val_masks = zip(*val_pairs)
    print(f"원본 {len(stems)}장 | 전체 페어 {len(pairs)} "
          f"| train {len(train_pairs)} | val {len(val_pairs)}")

    train_dataset = data_generator(list(train_images), list(train_masks), LABEL_PATH)
    val_dataset = data_generator(list(val_images), list(val_masks), LABEL_PATH)

    # [3] 현재 모델에 이어서 추가 학습 (fine-tuning)
    #     - 모델 구조/출력 채널은 기존 모델을 그대로 사용 (새로 만들지 않음)
    #     - 새 모델은 MODEL_PATH와 같은 폴더에 continued_model_{시각}.keras 로 저장됨
    continue_train(
        MODEL_PATH,
        train_dataset,
        val_dataset,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        patience=PATIENCE,
    )


if __name__ == '__main__':
    main()
