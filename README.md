# SHRC_GarbageDetection — DeepLabV3+ 학습 코드

드론 항공 이미지의 **쓰레기 영역 Semantic Segmentation** 모델을 학습하는 코드입니다.
**DeepLabV3+** 아키텍처(ImageNet 사전학습 **ResNet50** 백본)를 사용하며, 학습 산출물 `.keras` 모델은 서빙 시스템(`SHRC_GarbageEye`)의 `Modules/GarbageEye/models/`로 배포됩니다.

---

## 학습 파이프라인

```
[원본 이미지(.jpg) + RGB 레이블맵(.png)]
        │  augmenter.py — 오프라인 데이터 증강
        │  (좌우/상하 플립, 90·180·270° 회전, 랜덤 회전/쉬프트)
        ▼
[Aug_Images / Aug_SegmentationClass]
        │  data_generator — labelmap.txt 기반 RGB→클래스ID 변환
        │  (512×512 리사이즈, batch=16, 마스크는 Nearest Neighbor)
        ▼
[tf.data.Dataset]
        │  train() — 신규 학습  /  continue_train() — 추가 학습
        │  (tf.distribute.MirroredStrategy 다중 GPU)
        ▼
[saved_model/*.keras]  →  서빙 리포 Modules/GarbageEye/models/ 로 배포
```

---

## 프로젝트 구조

```
Train_DeepLab3/
├── main.py                       # 학습 진입점 (데이터 경로 지정, 학습 방식 선택)
├── Module/
│   ├── TrainMethods.py           # DeeplabV3 모델 정의, data_generator, train, continue_train
│   ├── TrainMethods_CorssVal.py  # K-Fold 교차검증 학습 버전
│   ├── augmenter.py              # 오프라인 데이터 증강 스크립트
│   ├── InferMethods.py           # 예측 시각화 + 정확도 평가
│   └── ProcessingMethods.py      # 데이터셋 검증/시각화 헬퍼
├── requirements.txt              # Python 의존성
└── README.md                     # 이 문서
```

---

## 모듈별 역할

| 파일 | 역할 |
|---|---|
| `main.py` | 학습 진입점. 데이터셋 경로/개수 지정 후 신규 학습(`train`) 또는 추가 학습(`continue_train`) 호출 |
| `Module/TrainMethods.py` | `DeeplabV3()` 모델 정의(ResNet50 + ASPP), `data_generator`, `train`, `continue_train` |
| `Module/TrainMethods_CorssVal.py` | 동일 파이프라인의 **K-Fold 교차검증** 버전 (`KFold`로 분할, 최고 성능 모델 저장) |
| `Module/augmenter.py` | 이미지 + 마스크 동시 증강 (학습 전 오프라인 전처리 단계) |
| `Module/InferMethods.py` | 예측 결과 9분할 시각화, 데이터셋/랜덤 샘플 픽셀 정확도 평가, 컬러맵 생성 |
| `Module/ProcessingMethods.py` | 이미지·마스크 개수 검증, 클래스 ID 범위 검증, 배치 시각화 |

---

## 모델 아키텍처 — DeepLabV3+

- **백본**: ResNet50 (`weights='imagenet'`, `include_top=False`)
  - 저수준 특징: `conv2_block3_2_relu`
  - 고수준 특징: `conv4_block6_2_relu`
- **ASPP** (Dilated Spatial Pyramid Pooling): dilation rate 1·6·12·18 + Global Average Pooling 브랜치
- **Decoder**: 저수준 특징과 융합(Concatenate) 후 bilinear 업샘플링
- **출력**: `Conv2D(num_classes, 1×1)` → `(512, 512, num_classes)` 로짓

---

## 학습 하이퍼파라미터

| 항목 | 값 |
|---|---|
| 입력 크기 | 512 × 512 × 3 (`pixel/127.5 - 1.0` 정규화) |
| 배치 크기 | 16 (`drop_remainder=True`) |
| 손실 함수 | `SparseCategoricalCrossentropy(from_logits=True)` |
| 옵티마이저 | Adam (`learning_rate=1e-5`) |
| 에폭 | 신규 학습 30, 추가 학습 기본 10~20 |
| 분산 전략 | `tf.distribute.MirroredStrategy` (다중 GPU 자동 감지 + memory growth) |
| 마스크 리사이즈 | Nearest Neighbor (범주형 레이블 보존) |

---

## 데이터셋 구조

`main.py`의 `DATA_DIR` 기준 디렉토리 구성:

```
{DATA_DIR}/
├── Images/                   # 원본 이미지 (.jpg)
├── SegmentationClass/        # RGB 레이블맵 (.png)
├── Aug_Images/               # 증강된 이미지 (augmenter.py 산출)
├── Aug_SegmentationClass/    # 증강된 레이블맵 (augmenter.py 산출)
└── labelmap.txt              # 클래스명:R,G,B 매핑
```

`labelmap.txt` 형식 (`#` 주석 및 빈 줄 허용):

```
# 클래스명:R,G,B
Background:0,0,0
Garbage:255,0,0
...
```

---

## 실행

```bash
pip install -r requirements.txt

# 1) (필요 시) 데이터 증강 — Images/SegmentationClass → Aug_*
python Module/augmenter.py

# 2) 학습 실행
#    main.py 상단의 DATA_DIR / LBAEL_PATH, train() vs continue_train() 선택을 수정 후
python main.py
```

### 신규 학습 vs 추가 학습 (`main.py`)

```python
# [1-1] 처음부터 학습
model = train(train_dataset, val_dataset)

# [1-2] 기존 .keras 모델에서 이어서 학습
continue_train(model_path, train_dataset, val_dataset, epochs=20)
```

학습 완료 시 `saved_model/`에 타임스탬프가 붙은 `.keras` 파일과 loss/accuracy 그래프가 저장됩니다.

---

## 산출물

| 산출물 | 설명 |
|---|---|
| `saved_model/model_{시각}.keras` | 신규 학습 모델 |
| `saved_model/continued_model_{시각}.keras` | 추가 학습 모델 |
| `saved_model/best_model_{시각}.keras` | K-Fold 최고 성능 모델 (`TrainMethods_CorssVal.py`) |
| `history.json` / `kfold_history_{시각}.json` | 학습 이력 |
| `Acc_Loss_{시각}.png` / `kfold_results_{시각}.png` | loss·accuracy 그래프 |

---

## 주의 사항

- `main.py`, `Module/augmenter.py` 등에 학습 서버 기준 **절대 경로**(`/home/keti_taehoon/...`)가 하드코딩되어 있습니다. 다른 환경에서 실행하려면 `DATA_DIR`, `LBAEL_PATH`, `IMG_DIR` 등을 수정해야 합니다.
- 모델 출력 클래스 수는 학습 시점의 `NUM_CLASSES` 설정을 따르며, 서빙 모델은 **6 클래스**(쓰레기 = index 1)입니다. 신규 학습 시 `Module/TrainMethods.py`의 `NUM_CLASSES`를 데이터셋에 맞게 확인하세요.
