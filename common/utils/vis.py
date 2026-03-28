import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_bboxes(img, bbox_list, color=(0, 255, 0), thickness=2):
    """Draw person bounding boxes. bbox_list: list of [x, y, w, h] (xy top-left, width/height)."""
    for bbox in bbox_list:
        x, y, w, h = [int(round(v)) for v in bbox[:4]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)
    return img


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    """이미지 크기에 맞춰 점/선 두께를 자동 스케일링합니다."""
    h, w = img.shape[:2]
    scale = max(1.0, min(h, w) / 400.0)
    radius = max(4, int(6 * scale))
    thickness = max(2, int(4 * scale))

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=thickness, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=radius, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=radius, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)


def vis_3d_skeleton_to_file(kpt_3d, kpt_3d_vis, kps_lines, path, title=None, view_elev=7, view_azim=62):
    """Draw 3D skeleton and save to file (no interactive plt.show). Same layout as MATLAB view(62,7)."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[0], c[1], c[2])) for c in colors]
    for l in range(len(kps_lines)):
        i1, i2 = kps_lines[l][0], kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])
        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=[colors[l]], marker="o")
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=[colors[l]], marker="o")
    ax.set_facecolor((1, 1, 1))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    xc, yc, zc = np.mean(kpt_3d, axis=0)
    margin = 1000
    ax.set_xlim(xc - margin, xc + margin)
    ax.set_ylim(zc - margin, zc + margin)
    ax.set_zlim(-yc - margin, -yc + margin)
    ax.view_init(elev=view_elev, azim=view_azim)
    if title:
        ax.set_title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    for l in range(len(kps_lines)):
        i1, i2 = kps_lines[l][0], kps_lines[l][1]
        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
            y = np.array([kpt_3d[n, i1, 1], kpt_3d[n, i2, 1]])
            z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])
            if kpt_3d_vis[n, i1, 0] > 0 and kpt_3d_vis[n, i2, 0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n, i1, 0] > 0:
                ax.scatter(kpt_3d[n, i1, 0], kpt_3d[n, i1, 2], -kpt_3d[n, i1, 1], c=[colors[l]], marker='o')
            if kpt_3d_vis[n, i2, 0] > 0:
                ax.scatter(kpt_3d[n, i2, 0], kpt_3d[n, i2, 2], -kpt_3d[n, i2, 1], c=[colors[l]], marker='o')
    ax.set_title(filename or '3D vis')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('-Y')
    plt.show()
    cv2.waitKey(0)


# Body-part colors for 3D skeleton (torso=yellow, arms=green/orange, legs=pink/blue)
_SKELETON_COLORS = [
    (1.0, 0.8, 0.0),   # 0 torso (0,7) (7,8)
    (0.2, 0.8, 0.4),   # 1 head (8,9)(9,10)
    (1.0, 0.6, 0.2),   # 2 L_arm upper (8,11)(11,12)
    (1.0, 0.8, 0.0),   # 3 L_arm lower (12,13)
    (0.2, 0.8, 0.4),   # 4 R_arm upper (8,14)(14,15)
    (1.0, 0.6, 0.2),   # 5 R_arm lower (15,16)
    (1.0, 0.4, 0.6),   # 6 L_leg upper (0,4)(4,5)
    (0.4, 0.7, 1.0),   # 7 L_leg lower (5,6)
    (1.0, 0.4, 0.6),   # 8 R_leg upper (0,1)(1,2)
    (0.4, 0.7, 1.0),   # 9 R_leg lower (2,3)
]
_EDGE_TO_COLOR_IDX = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9]  # per kps_lines index


def vis_3d_with_image_plane(img_bgr, kpt_3d, kpt_3d_vis, kps_lines, title="3D Pose", save_path=None):
    """2D 이미지를 3D 평면에 배치하고, 바닥 그리드 + 컬러 스켈레톤으로 시각화. save_path가 있으면 파일로 저장 후 종료."""
    from mpl_toolkits.mplot3d import art3d

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 좌표 범위 (camera space: X right, Y down, Z forward)
    x_all = kpt_3d[:, :, 0].flatten()
    z_all = kpt_3d[:, :, 2].flatten()
    y_all = -kpt_3d[:, :, 1].flatten()
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    z_min, z_max = float(np.min(z_all)), float(np.max(z_all))
    pad_x = max(200, (x_max - x_min) * 0.3)
    pad_z = max(200, (z_max - z_min) * 0.3)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    z_min, z_max = z_min - pad_z, z_max + pad_z
    y_min = min(0, float(np.min(y_all)) - 100)

    # 바닥 그리드 (흰색)
    gx = np.linspace(x_min, x_max, 12)
    gz = np.linspace(z_min, z_max, 12)
    for gxi in gx:
        ax.plot([gxi, gxi], [z_min, z_max], [y_min, y_min], color='white', linewidth=0.5, alpha=0.8)
    for gzi in gz:
        ax.plot([x_min, x_max], [gzi, gzi], [y_min, y_min], color='white', linewidth=0.5, alpha=0.8)

    # 2D 이미지를 뒷쪽 수직 평면에 배치 (X-Y 평면에서 Z=z_max)
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sw, sh = min(100, w), min(75, h)
    img_small = cv2.resize(img_rgb, (sw, sh))
    y_span = (x_max - x_min) * h / max(w, 1)
    xx = np.linspace(x_min, x_max, sw + 1)
    yy = np.linspace(y_min, y_min + y_span, sh + 1)
    xx, yy = np.meshgrid(xx, yy)
    zz = np.full_like(xx, z_max)
    fc = np.zeros((sh, sw, 4))
    for i in range(sh):
        for j in range(sw):
            fc[i, j, :3] = img_small[sh - 1 - i, j] / 255.0
            fc[i, j, 3] = 0.92
    ax.plot_surface(xx, zz, yy, facecolors=fc, rstride=1, cstride=1, shade=False)

    # 3D 스켈레톤 (컬러)
    for l in range(len(kps_lines)):
        i1, i2 = kps_lines[l][0], kps_lines[l][1]
        c = _SKELETON_COLORS[_EDGE_TO_COLOR_IDX[l] if l < len(_EDGE_TO_COLOR_IDX) else 0]
        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            if kpt_3d_vis[n, i1, 0] > 0 and kpt_3d_vis[n, i2, 0] > 0:
                x = np.array([kpt_3d[n, i1, 0], kpt_3d[n, i2, 0]])
                z = np.array([kpt_3d[n, i1, 2], kpt_3d[n, i2, 2]])
                y = np.array([-kpt_3d[n, i1, 1], -kpt_3d[n, i2, 1]])
                ax.plot(x, z, y, c=c, linewidth=2.5)
            if kpt_3d_vis[n, i1, 0] > 0:
                ax.scatter([kpt_3d[n, i1, 0]], [kpt_3d[n, i1, 2]], [-kpt_3d[n, i1, 1]], c=[c], s=25)
            if kpt_3d_vis[n, i2, 0] > 0:
                ax.scatter([kpt_3d[n, i2, 0]], [kpt_3d[n, i2, 2]], [-kpt_3d[n, i2, 1]], c=[c], s=25)

    ax.set_facecolor('#1a1a2e')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_zlim(y_min, None)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(title)
    ax.view_init(elev=15, azim=-70)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
    else:
        plt.show()
        cv2.waitKey(0)

