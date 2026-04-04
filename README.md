# MobileHumanPose

> PyTorch implementation of **MobileHumanPose: Toward real-time 3D human pose estimation in mobile devices** (CVPRW 2021).

**Last updated:** 2026-02-21

## Introduction

Official implementation of [MobileHumanPose (CVPRW 2021)](https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Choi_MobileHumanPose_Toward_Real-Time_3D_Human_Pose_Estimation_in_Mobile_Devices_CVPRW_2021_paper.html).  
파이프라인: **데이터셋 생성 → 학습 → 가중치 변환 → 데모 실행**

## Quick Start

```bash
pip install -e .
python scripts/generate_dummy_data.py   # 또는 python data/Dummy/generate_dummy_data.py
python -m src.train
python -m src.test --test_epoch 0
python demo/web_video_inference.py     # http://localhost:7860
```

**상세 사용법:** [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

## Directory

```
${ROOT}
├── assets/           # README, demo용 이미지/영상
│   └── videos/       # Web inference 출력 (pose_*.mp4)
├── common/           # 공용 기능 (backbone, logger, timer, base Trainer/Tester, vis)
├── data/             # 학습 데이터, DataLoader 정의 (Human36M, MuCo, MuPoTS, Dummy, MSCOCO, MPII)
├── demo/             # 시각화 및 추론: webcam 스트리밍(브라우저), 이미지/비디오 추론
├── docs/             # 사용법 문서 (모든 사용법은 여기 참조)
├── runs/             # 학습/평가 스크립트 (train.sh 등)
├── src/              # 모델 구현
│   ├── 3dpose_estimator/   # 3D keypoint (PoseNet)
│   ├── box_detector/       # person bbox
│   └── distance_estimator/ # 절대 z (root depth, RootNet)
├── scripts/          # generate_dummy_data, test_config_combinations 등
├── tool/             # 데이터 전처리 (선택)
├── vis/              # Legacy MATLAB 2D/3D; Python: common.utils.vis
└── output/           # log, model_dump, result, vis (생성됨)
```

## 3D visualization (GitHub 워크플로 준수, Python 기준)

[GitHub 3D visualization](https://github.com/SangbumChoi/MobileHumanPose#3d-visualization)과 동일한 순서: **이미지 이름 .txt → preds_2d/3d_kpt_$DB_NAME.mat 배치 → 시각화**. 시각화는 **전부 Python**으로 하며, 2D에는 **person bbox**와 스켈레톘, 3D는 **원근(view)** 을 반영해 그립니다.

1. **이미지 이름 목록(.txt)**  
   ```bash
   python scripts/export_coco_img_names.py    # → vis/coco_img_name.txt
   python scripts/export_mupots_img_names.py # → vis/mupots_img_name.txt
   ```
   데모 한 장만 쓸 때는 `demo/demo.py --save_mat`로 `output/result/coco_img_name.txt`와 .mat를 함께 생성할 수 있음.
2. **테스트 결과(.mat)**  
   `preds_2d_kpt_$DB_NAME.mat`, `preds_3d_kpt_$DB_NAME.mat`를 `output/result/` 등에 두고, 시각화 시 `--result_dir` 또는 `--mat_2d`/`--mat_3d`로 지정.
3. **시각화 (Python만 사용)**  
   ```bash
   python vis/draw_pose.py --dataset coco --result_dir output/result --root_path /path/to/images --save_dir vis/out
   python vis/draw_pose.py --dataset mupots --result_dir output/result --root_path /path/to/MultiPersonTestSet --save_dir vis/out
   ```
   - 2D: person bbox + 키포인트/스켈레톘  
   - 3D: 카메라 좌표 기준 키포인트, 원근 표현(view 고정)

MATLAB(`vis/single`, `vis/multi`의 `.m`)은 레거시이며, 동일 결과는 위 Python으로만 재현 가능합니다. 자세한 내용은 [vis/README.md](vis/README.md)를 참고하세요.

## Docs (사용법)

| 문서 | 내용 |
|------|------|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | 전체 파이프라인: 데이터셋 → 학습 → 변환 → 데모 |
| [docs/COREML_ONNX_CONVERSION.md](docs/COREML_ONNX_CONVERSION.md) | ONNX / CoreML 변환 |
| [docs/README.md](docs/README.md) | 문서 인덱스 |

## Agents (Cursor)

프로젝트에 포함된 subagent를 활용하면 작업 검증과 플로우 점검을 자동화할 수 있습니다.

| Agent | 역할 | 사용법 |
|-------|------|--------|
| **verifier** | 완료된 작업 검증, 구현 정상 동작 확인, 테스트 실행 후 통과/미완료 항목 보고 | 작업 완료 후 `verifier subagent로 검증해줘` 또는 Cursor에서 verifier 호출 |

### Verifier가 확인하는 흐름

1. **데이터셋** – `scripts/generate_dummy_data.py` 또는 `data/Dummy/generate_dummy_data.py`
2. **학습** – `python -m src.train`, `runs/train.sh`
3. **변환** – `python -m src.3dpose_estimator.export -f onnx/coreml -m <model>`
4. **데모** – `demo/demo.py`, `demo/web_video_inference.py`, `demo/video_inference.py`

Agent 설정: `.cursor/agents/verifier.md`

## Changelog

| Date | Change |
|------|--------|
| 2026-02-21 | Agents(verifier) 추가, docs 구조 정리, 디렉터리 역할 명시 |
| 2026-02-20 | Web video inference (Gradio), pre-commit+ruff, README overhaul |
| 2022-05-19 | Dummy dataloader for fast PoC |
| 2021-11-23 | Initial release |

## Citation

```bibtex
@InProceedings{Choi_2021_CVPR,
    author    = {Choi, Sangbum and Choi, Seokeon and Kim, Changick},
    title     = {MobileHumanPose: Toward Real-Time 3D Human Pose Estimation in Mobile Devices},
    booktitle = {CVPR Workshops},
    year      = {2021},
    pages     = {2328-2338}
}
```

## License

MIT
