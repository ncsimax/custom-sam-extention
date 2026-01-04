# Copyright (c) Meta Platforms, Inc. and affiliates.

# ### save outputs  :add 20251122-1753
import json
import numpy as np
import os
# ### move top "import sentence"


import argparse
import os
from glob import glob

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm



def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

# ### add : export rig
    export_dir = os.path.join(args.output_folder, "rig_info")

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    for image_path in tqdm(images_list):
        print(f"Processing {image_path} ...")
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        if len(outputs) == 0:
                print(f"  -> Warning: No person detected in {os.path.basename(image_path)}. Skipping.")
                continue  # next image

        def export_all_joints(outputs, output_dir, filename_prefix):
            """
            Exports joints to OBJ and JSON with a filename prefix to handle multiple images.
            """
            os.makedirs(output_dir, exist_ok=True)
    
            json_data = []

            for i, result in enumerate(outputs):
                # --- Extract Data ---
                kpts_70 = result['pred_keypoints_3d']
                if hasattr(kpts_70, 'cpu'):
                    kpts_70 = kpts_70.detach().cpu().numpy()
        
                joints_127 = result['pred_joint_coords']
                if hasattr(joints_127, 'cpu'):
                    joints_127 = joints_127.detach().cpu().numpy()

                # --- Export to OBJ ---
                # Include image filename in the output name
                name_base = f"{filename_prefix}_person_{i}"
                
                # Save 70 keypoints
                file_70 = os.path.join(output_dir, f"{name_base}_kpts70.obj")
                with open(file_70, 'w') as f:
                    for p in kpts_70:
                        f.write(f"v {p[0]} {p[1]} {p[2]}\n")
                
                # Save 127 joints
                file_127 = os.path.join(output_dir, f"{name_base}_joints127.obj")
                with open(file_127, 'w') as f:
                    for p in joints_127:
                        f.write(f"v {p[0]} {p[1]} {p[2]}\n")

                # --- Prepare for JSON ---
                person_entry = {
                    "image_file": filename_prefix,
                    "person_id": i,
                    "keypoints_70": kpts_70.tolist(),
                    "joints_127": joints_127.tolist()
                }
                json_data.append(person_entry)

            # --- Export to JSON (Append mode or unique file) ---
            # Saving separate JSON per image to avoid overwrite/complexity
            json_filename = os.path.join(output_dir, f"{filename_prefix}_joints.json")
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=4)
            
            print(f"  -> Saved JSON summary: {json_filename}")


	# OpenPoseの画像生成
        export_dir = os.path.join(args.output_folder, "exported_joints")
        file_name_only = os.path.splitext(os.path.basename(image_path))[0]
       
        export_mesh_and_joints(outputs,  estimator.faces, output_dir=export_dir, filename_prefix=file_name_only)
        export_pose_data(outputs, export_dir, file_name_only)
        export_rig_data(estimator, export_dir)

        # COCO format と BODY-25 format 両方を生成
        openpose_dir = os.path.join(args.output_folder, "openpose_results")
        export_openpose_images(outputs, image_path, openpose_dir)

        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        cv2.imwrite(
            f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img.astype(np.uint8),
        )


# write OBJ with RIG

def export_rig_data(estimator, output_dir):
    print("\n=== Exporting MHR Rig Data (Robust Ver.) ===")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # モジュールへのアクセス
        model_root = estimator.model
        char_module = model_root.head_pose.mhr.character_torch
        mesh_module = char_module.mesh
        skel_module = char_module.skeleton
        skin_module = char_module.linear_blend_skinning
    except AttributeError as e:
        print(f"Error accessing modules: {e}")
        return

    def to_np(tensor):
        return tensor.detach().cpu().numpy()

    # 基本データの取得
    rest_verts = to_np(mesh_module.rest_vertices)
    faces = to_np(mesh_module.faces)
    parents = to_np(skel_module.joint_parents)
    offsets = to_np(skel_module.joint_translation_offsets)
    prerots = to_np(skel_module.joint_prerotations)

    num_verts = rest_verts.shape[0] # 18439
    num_bones = parents.shape[0]    # 127

    print(f"  Target Shape -> Verts: {num_verts}, Bones: {num_bones}")

    # スキンウェイト探索
    weights = None
    # パラメータとバッファを全てリスト
    all_tensors = list(skin_module.named_parameters()) + list(skin_module.named_buffers())
    
    for name, tensor in all_tensors:
        shape = tensor.shape
        # サイズチェック
        if (shape[0] == num_verts and shape[1] == num_bones):
            print(f"  [FOUND] Weights found in variable: '{name}' (Shape: {shape})")
            weights = to_np(tensor)
            break
        elif (shape[0] == num_bones and shape[1] == num_verts):
            print(f"  [FOUND] Weights found (Transposed) in: '{name}' (Shape: {shape})")
            weights = to_np(tensor).T
            break
    
    if weights is None:
        print("  [ERROR] Skinning weights NOT found! Animation will not work.")
        # 空の配列を入れる（エラー回避）
        weights = np.array([])
    
    # 保存
    npz_path = os.path.join(output_dir, "mhr_rig_data_v2.npz")
    np.savez(
        npz_path,
        rest_vertices=rest_verts,
        faces=faces,
        joint_parents=parents,
        joint_offsets=offsets,
        joint_prerotations=prerots,
        skinning_weights=weights 
    )
    print(f"  -> Saved Rig Data: {npz_path}")


def export_pose_data(outputs, output_dir, filename_prefix):
    os.makedirs(output_dir, exist_ok=True)

    pose_data_list = []

    for i, result in enumerate(outputs):
        
        if 'body_pose_params' in result:
            pose_params = result['body_pose_params'] 
            if hasattr(pose_params, 'cpu'):
                pose_params = pose_params.detach().cpu().numpy()
            
            pose_data = {
                "person_id": i,
                "pose_params": pose_params.flatten().tolist(), 
                "shape": pose_params.shape 
            }
            pose_data_list.append(pose_data)

        json_path = os.path.join(output_dir, f"{filename_prefix}_person_{i}_pose.json")
        with open(json_path, 'w') as f:
            json.dump(pose_data_list, f, indent=4)
        
        print(f"  -> Saved Pose JSON: {json_path}")


def export_mesh_and_joints(outputs, faces, output_dir, filename_prefix):
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(faces, 'cpu'):
        faces = faces.detach().cpu().numpy()

    for i, result in enumerate(outputs):
        vertices = result['pred_vertices']
        if hasattr(vertices, 'cpu'):
            vertices = vertices.detach().cpu().numpy()

        kpts_70 = result['pred_keypoints_3d']
        if hasattr(kpts_70, 'cpu'):
            kpts_70 = kpts_70.detach().cpu().numpy()

        mesh_filename = os.path.join(output_dir, f"{filename_prefix}_person_{i}_mesh.obj")
        with open(mesh_filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"  -> Saved Mesh OBJ: {mesh_filename}")

        kpts_filename = os.path.join(output_dir, f"{filename_prefix}_person_{i}_kpts70.obj")
        with open(kpts_filename, 'w') as f:
            for p in kpts_70:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")


import cv2
import numpy as np
import os

def draw_openpose_format(img_size, keypoints, output_path):
    """
    OpenPose (COCO 18-keypoint format) image generator for SAM 3D Body.
    Based on analyzed index mapping from debug images.
    """
    canvas = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    # --- SAM 3D Body (MHR) to OpenPose COCO Mapping ---
    # 
    # OpenPose COCO Format:
    #  0:Nose, 1:Neck, 2:R-Sho, 3:R-Elb, 4:R-Wr, 5:L-Sho, 6:L-Elb, 7:L-Wr,
    #  8:R-Hip, 9:R-Knee, 10:R-Ank, 11:L-Hip, 12:L-Knee, 13:L-Ank,
    #  14:R-Eye, 15:L-Eye, 16:R-Ear, 17:L-Ear
    
    # MHR Indexメモ (0-based from pred_keypoints_2d  points 70? 127?)
    # Based on the debug image: 
    #  0=Nose, 1=R-Eye, 2=L-Eye, 3=R-Ear, 4=L-Ear
    #  68=L-Shoulder, 69=R-Shoulder
    #  66=L-Elbow, 65=R-Elbow
    #  41=L-Wrist, 64=R-Wrist
    #  Legs/Hips indices are inferred from standard SMPL-X topology if not visible in debug image.
    #  Usually: 1=L-Hip, 2=R-Hip, 4=L-Knee, 5=R-Knee, 7=L-Ankle, 8=R-Ankle (in SMPL order)
    #  BUT MHR output order might be different. Assuming 'pred_keypoints_2d' follows standard SMPL-X/OpenPose conventions partially.
    
    
    mhr_map = {
        0: 0,   # Nose
        69: 1, # Neck (計算で補完する方が綺麗だが、一旦パス)
        68: 2,  # R-Shoulder (画像より)
        8: 3,  # R-Elbow (画像より)
        41: 4,  # R-Wrist (画像より)
        67: 5,  # L-Shoulder (画像より)
        7: 6,  # L-Elbow (画像より)
        62: 7,  # L-Wrist (画像より)
        
        # 下半身 (SMPL順序を仮定)
        10: 8,   # R-Hip
        12: 9,   # R-Knee
        14: 10,  # R-Ankle
        9: 11,  # L-Hip
        11: 12,  # L-Knee
        13: 13,  # L-Ankle

        
        # 耳と目　OpenPose「14, 15, 16, 17」
        2: 14, # R-Eye
        1: 15, # L-Eye
        4: 16, # R-Ear
        3: 17, # L-Ear
        
        10: 8, 12: 9, 14: 10,  # Right Leg chain
        9: 11, 11: 12, 13: 13 # Left Leg chain
    }

    # Neck は 両肩の中点として計算
    # OpenPose Index 1
    neck_pos = None
    if 68 in keypoints and 69 in keypoints: # L-Sho, R-Sho indices
        l_sho = keypoints[68]
        r_sho = keypoints[69]
        # 座標が(0,0)でなければ
        if np.sum(l_sho) > 0 and np.sum(r_sho) > 0:
            neck_pos = (l_sho + r_sho) / 2
            # 描画リストに追加 (Neck = 1)
            x, y = int(neck_pos[0]), int(neck_pos[1])
            cv2.circle(canvas, (x, y), 4, (255, 85, 0), -1) # Neck Color

    # OpenPose Pairs (Start, End, Color)
    pairs = [
        (1, 2, (255, 85, 0)),   (1, 5, (255, 255, 0)),  # Neck -> Shoulders
        (2, 3, (255, 170, 0)),  (3, 4, (255, 255, 0)),  # R-Arm
        (5, 6, (170, 255, 0)),  (6, 7, (85, 255, 0)),   # L-Arm
        (1, 8, (0, 255, 85)),   (8, 9, (0, 255, 170)),  (9, 10, (0, 255, 255)), # R-Leg
        (1, 11, (0, 170, 255)), (11, 12, (0, 85, 255)), (12, 13, (0, 0, 255)),  # L-Leg
        (0, 1, (255, 0, 0)),    # Nose -> Neck
        (0, 14, (255, 0, 170)), (14, 16, (255, 0, 85)), # R-Eye, R-Ear
        (0, 15, (255, 0, 170)), (15, 17, (255, 0, 85))  # L-Eye, L-Ear (Color approx)
    ]

    # Color Palette (BGR) for 18 points
    colors = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
        (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
        (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
        (255, 0, 255), (255, 0, 170), (255, 0, 85)
    ]

    # 座標辞書の作成
    op_points = {} 

    # 通常マッピング
    for mhr_idx, op_idx in mhr_map.items():
        if mhr_idx < len(keypoints):
            pt = keypoints[mhr_idx]
            x, y = int(pt[0]), int(pt[1])
            if x > 0 and y > 0: # (0,0)は未検出とみなす
                op_points[op_idx] = (x, y)
                # 描画
                cv2.circle(canvas, (x, y), 4, colors[op_idx], -1)
    
    # 計算したNeckを追加
    if neck_pos is not None:
        op_points[1] = (int(neck_pos[0]), int(neck_pos[1]))

    # 線を描画
    for p1, p2, color in pairs:
        if p1 in op_points and p2 in op_points:
            cv2.line(canvas, op_points[p1], op_points[p2], color, 3)

    cv2.imwrite(output_path, canvas)
    print(f"  -> Saved OpenPose Image: {output_path}")



def draw_openpose_full_body_hand(img_size, keypoints, output_path):
    """
    OpenPose BODY_25 + Hands (Corrected Mapping for SAM 3D Body)
    """
    canvas = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    # ---------------------------------------------------------
    # 1. BODY_25 Mapping (Corrected based on debug image)
    # ---------------------------------------------------------
    # OpenPose Index:
    # 0:Nose, 1:Neck, 2:RSho, 3:RElb, 4:RWr, 5:LSho, 6:LElb, 7:LWr
    # 8:MidHip, 9:RHip, 10:RKnee, 11:RAnk, 12:LHip, 13:LKnee, 14:LAnk
    # 19:LBigToe, 20:LSmallToe, 21:LHeel, 22:RBigToe, 23:RSmallToe, 24:RHeel
    
    # MHRインデックス
    # Right=人物の右側, Left=左側
    body_map = {
        0: 0,   # Nose
        
        # Right Arm
        68: 2,  # R-Shoulder
        66: 3,  # R-Elbow
        41: 4,  # R-Wrist (画像で確認: 向かって左の手袖口が41)
        
        # Left Arm
        69: 5,  # L-Shoulder
        65: 6,  # L-Elbow
        63: 7,  # L-Wrist (画像で確認: 63が袖口付近)
        
        # Right Leg
        10: 9,  # R-Hip
        12: 10, # R-Knee
        14: 11, # R-Ankle
        19: 22, # R-BigToe
        20: 23, # R-SmallToe
        18: 24, # R-Heel (推定)

        # Left Leg
        9: 12,  # L-Hip
        11: 13, # L-Knee
        13: 14, # L-Ankle
        15: 19, # L-BigToe
        16: 20, # L-SmallToe
        17: 21, # L-Heel (推定)
        
        # Face
        1: 15, 2: 16, 3: 17, 4: 18
    }

    # 座標取得 & (0,0)除外
    body_points = {}
    for mhr, op in body_map.items():
        if mhr < len(keypoints):
            pt = keypoints[mhr]
            # 座標が(0,0)や極端に小さい場合は描画しない
            if pt[0] > 1 and pt[1] > 1: 
                body_points[op] = (int(pt[0]), int(pt[1]))

    # 補完: Neck (1)
    if 2 in body_points and 5 in body_points:
        body_points[1] = ((body_points[2][0]+body_points[5][0])//2, (body_points[2][1]+body_points[5][1])//2)
    
    # 補完: MidHip (8)
    if 9 in body_points and 12 in body_points:
        body_points[8] = ((body_points[9][0]+body_points[12][0])//2, (body_points[9][1]+body_points[12][1])//2)

    # ---------------------------------------------------------
    # 2. Hand Mapping
    # ---------------------------------------------------------
    
    # 右手 (Wrist=41 の先)
    r_hand_indices = [
        41, # Wrist
        24, 23, 28, 22, # Thumb
        40, 36, 32, 21, # Index
        39, 35, 34, 33, # Middle
        38, 37, 31, 30, # Ring
        27, 26, 25, 29  # Pinky
    ]

    # 左手 (Wrist=63 の先)
    l_hand_indices = [
        63, # Wrist
        49, 44, 48, 43, # Thumb
        53, 45, 52, 42, # Index
        57, 50, 51, 47, # Middle
        62, 56, 55, 54, # Ring
        61, 60, 59, 58  # Pinky
    ]

    hands_points = {'right': {}, 'left': {}}

    # 右手抽出
    for i, idx in enumerate(r_hand_indices):
        if idx < len(keypoints):
            pt = keypoints[idx]
            if pt[0] > 1 and pt[1] > 1:
                hands_points['right'][i] = (int(pt[0]), int(pt[1]))
    
    # 左手抽出
    for i, idx in enumerate(l_hand_indices):
        if idx < len(keypoints):
            pt = keypoints[idx]
            if pt[0] > 1 and pt[1] > 1:
                hands_points['left'][i] = (int(pt[0]), int(pt[1]))

    # 3. 描画 (Colors & Lines)
    # ---------------------------------------------------------
    
    # --- Body Colors (OpenPose Standard - BGR) ---
    # 関節点の色
    body_colors = [
        (0, 0, 255), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), # 0-4
        (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),  # 5-9
        (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),  # 10-14
        (255, 0, 255), (255, 0, 170), (255, 0, 85), (85, 255, 255), (170, 255, 255), # 15-19
        (255, 255, 170), (255, 255, 85), (255, 170, 255), (255, 85, 255), (85, 85, 255) # 20-24
    ]
    
    # 骨の接続ペアと色 (Start, End, Color) カラフルに
    body_pairs_colors = [
        (1, 8, (0, 100, 255)), (1, 2, (0, 100, 255)), (1, 5, (0, 100, 255)), # Torso (Orange)
        (2, 3, (0, 255, 255)), (3, 4, (0, 255, 255)), # R-Arm (Yellow)
        (5, 6, (0, 255, 0)),   (6, 7, (0, 255, 0)),   # L-Arm (Green)
        (8, 9, (0, 200, 255)), (9, 10, (0, 200, 255)), (10, 11, (0, 200, 255)), # R-Leg (Light Orange)
        (8, 12, (255, 100, 0)),(12, 13, (255, 100, 0)), (13, 14, (255, 100, 0)),# L-Leg (Blue)
        (1, 0, (255, 0, 100)), (0, 15, (255, 0, 100)), (15, 17, (255, 0, 100)), # Face (Purple)
        (0, 16, (255, 0, 100)), (16, 18, (255, 0, 100)),
        (14, 19, (255, 100, 0)), (19, 20, (255, 100, 0)), (14, 21, (255, 100, 0)), # L-Foot
        (11, 22, (0, 200, 255)), (22, 23, (0, 200, 255)), (11, 24, (0, 200, 255))  # R-Foot
    ]

    # Draw Body Lines
    for p1, p2, color in body_pairs_colors:
        if p1 in body_points and p2 in body_points:
            cv2.line(canvas, body_points[p1], body_points[p2], color, 3)
            
    # Draw Body Points
    for idx, pt in body_points.items():
        if idx < len(body_colors):
            cv2.circle(canvas, pt, 4, body_colors[idx], -1)

    # --- Hand Drawing ---
    hand_pairs = [
        (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8),
        (0,9), (9,10), (10,11), (11,12), (0,13), (13,14), (14,15), (15,16),
        (0,17), (17,18), (18,19), (19,20)
    ]
    
    for side in ['right', 'left']:
        points = hands_points[side]
        # Line Color: Right=Blue, Left=Green (OpenPose conventionish)
        line_color = (0, 0, 255) if side == 'left' else (0, 0, 255)

        for p1, p2 in hand_pairs:
            if p1 in points and p2 in points:
                cv2.line(canvas, points[p1], points[p2], line_color, 2)

        for idx, pt in points.items():
            # 指の関節ごとに色を変える
            h_color = body_colors[idx % len(body_colors)]
            cv2.circle(canvas, pt, 3, h_color, -1)

    cv2.imwrite(output_path, canvas)
    print(f"  -> Saved OpenPose Full Image: {output_path}")



def export_openpose_images(outputs, image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 元画像のサイズ取得 (Pillowなどで読み込んでいる想定)
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size
    
    filename = os.path.splitext(os.path.basename(image_path))[0]

    for i, result in enumerate(outputs):
        # 2Dキーポイント取得
        kpts_2d = result['pred_keypoints_2d']
        if hasattr(kpts_2d, 'cpu'):
            kpts_2d = kpts_2d.detach().cpu().numpy()
            
        # 出力 COCO format
        save_path = os.path.join(output_dir, f"{filename}_person_{i}_openpose.png")
        draw_openpose_format((width, height), kpts_2d, save_path)
        # 出力 BODY_25 full (with fingers) format
        save_path = os.path.join(output_dir, f"{filename}_person_{i}_openpose_full.png")
        draw_openpose_full_body_hand((width, height), kpts_2d, save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    args = parser.parse_args()

    main(args)



 