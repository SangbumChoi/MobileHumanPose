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
