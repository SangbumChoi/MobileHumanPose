# vis

- **2D/3D 시각화**: 이 레포에서는 **Python** 기준으로 `common.utils.vis`를 사용합니다.
  - `vis_keypoints(img, kps, skeleton)` — 2D 스켈레톤 오버레이
  - `vis_3d_skeleton(kpt_3d, kpt_3d_vis, skeleton)` — 단일 3D 스켈레톤
  - `vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, skeleton)` — 다인 3D 스켈레톤  
  demo, test, 데이터셋 평가 코드에서 위 함수들을 사용합니다.

## 3D visualization (MATLAB) — README와 동일 워크플로

1. **이미지 이름 목록(.txt)**  
   `$DB_NAME` 자리에는 데이터셋 이름(`coco`, `mupots`)을 넣습니다.
   ```bash
   python scripts/export_coco_img_names.py    # → vis/coco_img_name.txt
   python scripts/export_mupots_img_names.py   # → vis/mupots_img_name.txt
   ```
   (기존 `.mat` 결과에서 키만 뽑으려면 `--mat output/result/preds_2d_kpt_coco.mat` 등으로 지정.)

2. **테스트 결과(.mat) 배치**  
   `preds_2d_kpt_$DB_NAME.mat`, `preds_3d_kpt_$DB_NAME.mat`를 **vis/single** 또는 **vis/multi** 폴더에 둡니다.  
   (테스트 시 `output/result/`에 생성되므로, 여기로 복사하면 됩니다.)

3. **MATLAB 실행**  
   **vis/single** 또는 **vis/multi**를 현재 폴더로 연 뒤:
   ```matlab
   draw_3Dpose_coco    % COCO
   draw_3Dpose_mupots % MuPoTS
   ```

- **경로 설정**: `draw_3Dpose_coco.m` / `draw_3Dpose_mupots.m` 안의 `root_path`(원본 이미지 디렉터리), `save_path`(저장 디렉터리)를 사용 환경에 맞게 수정해야 합니다.

## 3D visualization (Python) — 기본 시각화 (GitHub 워크플로 준수)

**vis/draw_pose.py**는 GitHub 3D visualization과 동일한 입력(.mat + `*_img_name.txt`)을 사용하며, **전부 Python**으로 2D/3D를 그립니다.

- **2D**: person **bbox** + 키포인트/스켈레톘 (bbox는 키포인트 min/max로 추정하거나 demo에서 그린 것과 동일한 방식).
- **3D**: 카메라 좌표 기준, **원근(view)** 고정(`elev=7`, `azim=62`)으로 저장.

```bash
# COCO (테스트 결과 또는 demo --save_mat 출력 사용)
python vis/draw_pose.py --dataset coco --result_dir output/result --root_path /path/to/images --save_dir vis/out

# MuPoTS
python vis/draw_pose.py --dataset mupots --result_dir output/result --root_path /path/to/MultiPersonTestSet --save_dir vis/out
```

- **.mat 생성**: `python demo/demo.py -i image.jpg -m ... --save_mat --result_dir output/result` 로 `preds_2d_kpt_coco.mat`, `preds_3d_kpt_coco.mat`, `coco_img_name.txt` 생성 가능 (MSCOCO.evaluate와 동일 포맷).
- `--img_name_txt`: 기본값 `vis/coco_img_name.txt` 또는 `vis/mupots_img_name.txt`.
- `--mat_2d` / `--mat_3d`: .mat 경로 직접 지정 시 사용.
- 출력: `--save_dir` 아래 `*_2d.jpg`(bbox+스켈레톘), `*_3d.jpg`(3D 스켈레톤).
