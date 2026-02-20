# vis

- **2D/3D 시각화**: 이 레포에서는 **Python** 기준으로 `common.utils.vis`를 사용합니다.
  - `vis_keypoints(img, kps, skeleton)` — 2D 스켈레톤 오버레이
  - `vis_3d_skeleton(kpt_3d, kpt_3d_vis, skeleton)` — 단일 3D 스켈레톤
  - `vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, skeleton)` — 다인 3D 스켈레톤  
  demo, test, 데이터셋 평가 코드에서 위 함수들을 사용합니다.

- **이미지 이름 목록 추출** (COCO/MuPoTS): MATLAB용 텍스트 파일이 필요하면 **scripts**에서 실행하세요.
  - `python scripts/export_coco_img_names.py` → `output/vis/coco_img_name.txt`
  - `python scripts/export_mupots_img_names.py` → `output/vis/mupots_img_name.txt`

- **MATLAB (레거시)**  
  `single/`, `multi/`의 `.m` 파일은 예전 실험용 2D/3D 그리기 스크립트입니다.  
  MATLAB이 있을 때만 사용하며, preds_2d_kpt_*.mat / preds_3d_kpt_*.mat 및 위에서 만든 `*_img_name.txt`를 사용합니다.
