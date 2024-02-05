import pickle
import numpy as np
import cv2
import json
import glob
import subprocess
import shutil
import torch
from scipy.spatial.transform import Rotation
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from smplpytorch.display_utils import display_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
import datetime
import open3d as o3d
import scipy.spatial.transform as spt
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull, KDTree
from scipy.optimize import least_squares
from numpy.linalg import inv, norm
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import glob
import os
import pickle
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.nn import functional as F
import json
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init
import datetime
import random
import itertools
import datetime
import torch.nn as nn
from pytorch_lightning import Trainer
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import scipy.spatial.transform as spt
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import glob
import os
import pickle
import wandb
import numpy as np
import pytorch_lightning as pl
from torch.nn import functional as F
import json
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.init as init
import datetime
import random
import itertools
import datetime
import torch.nn as nn
from pytorch_lightning import Trainer
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import scipy.spatial.transform as spt
import pickle
from scipy.spatial import cKDTree
from quad_mesh_simplify import simplify_mesh


best_val_loss = float("inf")
best_params = None

# Set the WANDB_CACHE_DIR environment variable
os.environ["WANDB_CACHE_DIR"] = "/scratch_net/biwidl307/lgermano/crossvit/wandb/cache"

LEARNING_RATE = 0.001
BATCH_SIZE = 16
DROPOUT_RATE = 0
LAYER_SIZES_1 = [256, 256, 256]
LAYER_SIZES_3 = [64, 128, 256, 128, 64]
# Initialize the input to 24 joints
INITIAL_OBJ_PRED = torch.rand((BATCH_SIZE, 24))
ALPHA = 10
LAMBDA_1 = 100
LAMBDA_2 = 0.01
EPOCHS = 25
L = 4
input_dim = 72 * L * 2
output_stage1 = 256
input_stage2 = 512
output_stage2 = 3
input_stage3 = 6  # 2 * (3 * wandb.config.L * 2)
output_dim = 3

# Initialize wandb with hyperparameters
wandb.init(
    project="MLP",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "MLP",
        "dataset": "BEHAVE",
        "batch_size": BATCH_SIZE,
        "dropout_rate": DROPOUT_RATE,
        "layer_sizes_1": LAYER_SIZES_1,
        "layer_sizes_3": LAYER_SIZES_3,
        "alpha": ALPHA,
        "lambda_1": LAMBDA_1,
        "lambda_2": LAMBDA_2,
        "L": L,
        "epochs": EPOCHS,
    },
    mode="disabled",
)


def load_config(camera_id, base_path, Date="Date01"):
    config_path = os.path.join(base_path, "calibs", Date, "config", str(camera_id), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def project_mesh_on_image(
    img,
    projected_verts,
    faces,
    obj_projected_verts,
    obj_faces,
    candidate_obj_projected_verts,
    candidate_faces_np,
    projected_selected_joints,
    fraction=1.0,
    transparency=0.4,
    scale_factor=1.0,
):
    ##print("Start of project_mesh_on_image function.")
    ##print(f"Number of projected_verts_smpl: {len(projected_verts)}")

    # Convert faces to a list if it's a numpy array
    if isinstance(faces, np.ndarray):
        faces = faces.tolist()

    ##print(f"Original number of faces: {len(faces)}")

    # Randomly sample a fraction of the faces
    sampled_faces = random.sample(faces, int(len(faces) * fraction))
    ##print(f"Number of sampled faces: {len(sampled_faces)}")

    # Randomly sample a fraction of the obj faces
    obj_sampled_faces = obj_faces
    candidate_obj_sampled_faces = candidate_faces_np
    # random.sample(obj_faces, int(len(obj_faces) * fraction))
    ###print(f"Number of obj sampled faces: {len(obj_sampled_faces)}")

    # Resize the image based on the scale_factor
    img_height, img_width = img.shape[:2]
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)
    img = cv2.resize(img, (new_width, new_height))

    triangles2D = []
    for face in sampled_faces:
        triangle3D = [projected_verts[face[0]], projected_verts[face[1]], projected_verts[face[2]]]

        # Clamping the projected vertices
        # triangle3D_clamped = [(min(max(int(v[0]), 0), new_width-1), min(max(int(v[1]), 0), new_height-1)) for v in triangle3D]

        triangle2D = np.array([triangle3D], dtype=np.int32)
        triangles2D.append(triangle2D)

    obj_triangles2D = []
    for face in obj_sampled_faces:
        obj_triangle3D = [obj_projected_verts[face[0]], obj_projected_verts[face[1]], obj_projected_verts[face[2]]]

        # Clamping the projected vertices
        # obj_triangle3D_clamped = [(min(max(int(v[0]), 0), new_width-1), min(max(int(v[1]), 0), new_height-1)) for v in obj_triangle3D]

        obj_triangle2D = np.array([obj_triangle3D], dtype=np.int32)
        obj_triangles2D.append(obj_triangle2D)

    candidate_obj_triangles2D = []
    for face in candidate_obj_sampled_faces:
        candidate_obj_triangle3D = [
            candidate_obj_projected_verts[face[0]],
            candidate_obj_projected_verts[face[1]],
            candidate_obj_projected_verts[face[2]],
        ]

        # Clamping the projected vertices
        # obj_triangle3D_clamped = [(min(max(int(v[0]), 0), new_width-1), min(max(int(v[1]), 0), new_height-1)) for v in obj_triangle3D]

        candidate_obj_triangle2D = np.array([candidate_obj_triangle3D], dtype=np.int32)
        candidate_obj_triangles2D.append(candidate_obj_triangle2D)

    # Electric blue color
    electric_blue = (255, 0, 0)

    green = (0, 255, 0)

    # Red color
    red = (0, 0, 255)

    # Create a black image (mask) of the same size to draw the triangles
    mask = np.zeros_like(img)
    obj_mask = np.zeros_like(img)
    candidate_obj_mask = np.zeros_like(img)

    # Draw triangles on the mask
    for triangle in triangles2D:
        cv2.drawContours(mask, triangle, -1, electric_blue, -1)

    for triangle in obj_triangles2D:
        cv2.drawContours(obj_mask, triangle, -1, red, -1)

    for triangle in candidate_obj_triangles2D:
        cv2.drawContours(candidate_obj_mask, triangle, -1, green, -1)

    x = 0  # the amount you want to translate towards the right
    H = np.array([[1, 0, x], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    # Warp the mask using the homography
    aligned_mask = cv2.warpPerspective(mask, H, (img.shape[1], img.shape[0]))
    obj_aligned_mask = cv2.warpPerspective(obj_mask, H, (img.shape[1], img.shape[0]))
    candidate_obj_aligned_mask = cv2.warpPerspective(candidate_obj_mask, H, (img.shape[1], img.shape[0]))

    ########

    # # Read the background.ply and extract vertices
    # background_pcd = o3d.io.read_point_cloud("/scratch_net/biwidl307_second/lgermano/behave/calibs/Date01/background/background.ply")
    # background_vertices = np.asarray(background_pcd.points)

    # # Gray color for background points
    # gray = (0, 255, 0)

    # base_path = "/scratch_net/biwidl307_second/lgermano/behave"
    # intrinsics, distortion_coeffs = load_intrinsics_and_distortion(1, base_path)

    # alpha = transparency * 0.5  # Adjust this value. Smaller values will make the gray more transparent.

    # for vert in background_vertices:
    #     u, v = project_to_image(vert, intrinsics, distortion_coeffs)
    #     if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
    #         blended_color = tuple([int(gray_val * alpha + img[v, u][i] * (1 - alpha)) for i, gray_val in enumerate(gray)])
    #         img[v, u] = blended_color

    #######

    img = (1 - transparency) * img + transparency * (aligned_mask + obj_aligned_mask + candidate_obj_aligned_mask)

    # #  Plotting the joint points
    # def var_to_label(var_name):
    #     """Converts a variable name to a readable label."""
    #     words = var_name.split('_')
    #     label = ' '.join([word.capitalize() for word in words])
    #     return label

    # # Assuming joint_points is a list of 2D coordinates for the selected joints.
    # joint_points = projected_selected_joints

    # # Colors for each joint
    # joint_colors = [
    #     (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),  # Green, Yellow, Magenta, Cyan
    #     (255, 0, 0), (0, 0, 255), (128, 128, 128), (0, 128, 128),  # Red, Blue, Gray, Teal
    #     (128, 0, 128), (128, 128, 0), (0, 128, 0), (128, 0, 0),    # Purple, Olive, Green, Maroon
    #     (64, 0, 128), (0, 64, 128), (128, 64, 0), (0, 128, 64),    # Additional colors
    #     (64, 128, 0), (128, 0, 64), (255, 128, 0), (0, 128, 255),
    #     (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255)
    # ]

    # # Names for each joint (converted from variable names)
    # selected_joints = projected_selected_joints
    # selected_joint_names = [
    #     'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    #     'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    #     'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    #     'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    # ]
    # selected_joint_colors = joint_colors[:len(selected_joints)]

    # # Drawing the selected joints on the image
    # for point, color in zip(joint_points, selected_joint_colors):
    #     x, y = point  # Assuming the z-value is the depth and is not needed here
    #     cv2.circle(img, (int(x), int(y)), radius=7, color=color, thickness=-1)

    # # Adding a legend on the top-right corner
    # start_y = 10
    # start_x = img.shape[1] - 200  # Adjusting the starting point to fit longer joint names

    # for i, (name, color) in enumerate(zip(selected_joint_names, selected_joint_colors)):
    #     cv2.circle(img, (start_x, start_y + i*30), radius=7, color=color, thickness=-1)
    #     cv2.putText(img, name, (start_x + 20, start_y + i*30 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # ##print("End of project_mesh_on_image function.")
    return img


def render_smpl(transformed_pose, transformed_trans, betas, intrinsics_cam, distortion_cam, img):
    ##print("Start of render_smpl function.")

    batch_size = 1
    ##print(f"batch_size: {batch_size}")

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0, gender="male", model_root="/scratch_net/biwidl307/lgermano/smplpytorch/smplpytorch/native/models/"
    )
    ##print("SMPL_Layer created.")

    # Process pose parameters
    pose_params_start = torch.tensor(transformed_pose[:3], dtype=torch.float32)
    pose_params_rest = torch.tensor(transformed_pose[3:72], dtype=torch.float32)
    pose_params_rest[-6:] = 0
    pose_params = torch.cat([pose_params_start, pose_params_rest]).unsqueeze(0).repeat(batch_size, 1)
    ##print(f"pose_params shape: {pose_params.shape}")

    shape_params = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    ##print(f"shape_params shape: {shape_params.shape}")

    obj_trans = torch.tensor(transformed_trans, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    ##print(f"obj_trans shape: {obj_trans.shape}")

    # # Process obj parameters
    # obj_obj_trans = torch.tensor(transformed_obj_trans, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    # ##print(f"Obj_trans shape: {obj_obj_trans.shape}")

    # obj_pose_params = torch.tensor(transformed_obj_pose, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    # ##print(f"Obj_pose shape: {obj_trans.shape}")

    # GPU mode
    cuda = torch.cuda.is_available()
    ##print(f"CUDA available: {cuda}")
    device = torch.device("cuda:0" if cuda else "cpu")
    ##print(f"Device: {device}")

    pose_params = pose_params.to(device)
    shape_params = shape_params.to(device)
    obj_trans = obj_trans.to(device)
    smpl_layer = smpl_layer.to(device)
    # obj_obj_trans = obj_obj_trans.to(device)
    # obj_pose_params = obj_pose_params.to(device)
    ##print("All tensors and models moved to device.")

    # Forward from the SMPL layer
    verts, J = smpl_layer(pose_params, th_betas=shape_params, th_trans=obj_trans)

    ##print(J.shape)
    ##print(verts.shape)

    J = J.squeeze(0)
    # verts = verts.squeeze(0)

    ##print(J.shape)
    ###print(verts.shape)

    # Extracting joints from SMPL skeleton

    #  0: 'pelvis',
    #  1: 'left_hip',
    #  2: 'right_hip',
    #  3: 'spine1',
    #  4: 'left_knee',
    #  5: 'right_knee',
    #  6: 'spine2',
    #  7: 'left_ankle',
    #  8: 'right_ankle',
    #  9: 'spine3',
    # 10: 'left_foot',
    # 11: 'right_foot',
    # 12: 'neck',
    # 13: 'left_collar',
    # 14: 'right_collar',
    # 15: 'head',
    # 16: 'left_shoulder',
    # 17: 'right_shoulder',
    # 18: 'left_elbow',
    # 19: 'right_elbow',
    # 20: 'left_wrist',
    # 21: 'right_wrist',
    # 22: 'left_hand',
    # 23: 'right_hand'

    # Extracting joints from SMPL skeleton

    # Defining all the joints
    pelvis = J[0]
    left_hip = J[1]
    right_hip = J[2]
    spine1 = J[3]
    left_knee = J[4]
    right_knee = J[5]
    spine2 = J[6]
    left_ankle = J[7]
    right_ankle = J[8]
    spine3 = J[9]
    left_foot = J[10]
    right_foot = J[11]
    neck = J[12]
    left_collar = J[13]
    right_collar = J[14]
    head = J[15]
    left_shoulder = J[16]
    right_shoulder = J[17]
    left_elbow = J[18]
    right_elbow = J[19]
    left_wrist = J[20]
    right_wrist = J[21]
    left_hand = J[22]
    right_hand = J[23]

    # #Creating a list with all joints
    selected_joints = [
        pelvis,
        left_hip,
        right_hip,
        spine1,
        left_knee,
        right_knee,
        spine2,
        left_ankle,
        right_ankle,
        spine3,
        left_foot,
        right_foot,
        neck,
        left_collar,
        right_collar,
        head,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_hand,
        right_hand,
    ]

    # Creating a list with all joints
    # selected_joints = [pelvis, head, left_shoulder, right_shoulder, left_hand, right_hand]

    # # # Creating the meaningful subset list for predicting backpack position
    # selected_joints = [
    #       left_foot, right_foot, left_hand, right_hand
    # ]
    # selected_joints = [pelvis, left_knee, right_knee, spine2, left_ankle, right_ankle, spine3,
    #                 left_foot, right_foot, head, left_shoulder, right_shoulder, left_hand, right_hand]

    ###print(f"verts shape: {verts.shape}, Jtr shape: {Jtr.shape}")
    verts = verts.cpu()  # Move verts to CPU for subsequent operations
    ##print("verts moved to CPU.")
    projected_verts = [project_to_image(vert, intrinsics_cam, distortion_cam) for vert in verts[0].detach().numpy()]
    ##print(f"Number of projected_verts: {len(projected_verts)}")

    verts = verts.squeeze(0).cpu().numpy()

    return selected_joints, verts, projected_verts, smpl_layer.th_faces.cpu().numpy()


# def transform_smpl_to_camera_frame(pose, trans, cam_params):
#     # Convert axis-angle representation to rotation matrix
#     R_w = Rotation.from_rotvec(pose[:3]).as_matrix()

#     # Build transformation matrix of mesh in world coordinates
#     T_mesh = np.eye(4)
#     T_mesh[:3, :3] = R_w
#     T_mesh[:3, 3] = trans

#     # Extract rotation and translation of camera from world coordinates
#     R_w_c = np.array(cam_params['rotation']).reshape(3, 3)
#     t_w_c = np.array(cam_params['translation']).reshape(3,)

#     # Build transformation matrix of camera in world coordinates
#     T_cam = np.eye(4)
#     T_cam[:3, :3] = R_w_c
#     T_cam[:3, 3] = t_w_c

#     T_cam = T_cam.astype(np.float64)
#     T_mesh = T_mesh.astype(np.float64)
#     T_mesh_in_cam = np.linalg.inv(T_cam) @ T_mesh

#     # Extract transformed pose and translation of mesh in camera coordinate frame
#     transformed_pose = Rotation.from_matrix(T_mesh_in_cam[:3, :3]).as_rotvec().flatten()
#     transformed_pose = np.concatenate([transformed_pose, pose[3:]]).flatten()
#     transformed_trans = T_mesh_in_cam[:3, 3].flatten()

#     return transformed_pose, transformed_trans


def plot_obj_in_camera_frame(obj_pose, obj_trans, obj_template_path):
    # Load obj template
    # object_template = "/scratch_net/biwidl307_second/lgermano/behave/objects/stool/stool.obj"
    object_mesh = o3d.io.read_triangle_mesh(obj_template_path)
    object_vertices = np.asarray(object_mesh.vertices)

    # Debug: ##print object vertices before any transformation
    ##print("Object vertices before any transformation: ", object_vertices)

    # Compute the centroid of the object
    centroid = np.mean(object_vertices, axis=0)

    # Translate all vertices such that the object's centroid is at the origin
    object_vertices = object_vertices - centroid

    # Convert axis-angle representation to rotation matrix
    R_w = Rotation.from_rotvec(obj_pose).as_matrix()

    # Build transformation matrix of mesh in world coordinates
    T_mesh = np.eye(4)
    T_mesh[:3, :3] = R_w  # No rotation applied, keeping it as identity matrix
    T_mesh[:3, 3] = obj_trans

    # Debug: Verify T_mesh
    # ##print("T_mesh: ", T_mesh)

    # # Extract rotation and translation of camera from world coordinates
    # R_w_c = np.array(cam_params['rotation']).reshape(3, 3)
    # t_w_c = np.array(cam_params['translation']).reshape(3,)

    # # Build transformation matrix of camera in world coordinates
    # T_cam = np.eye(4)
    # T_cam[:3, :3] = R_w_c
    # T_cam[:3, 3] = t_w_c

    # # Debug: Verify T_cam
    # ##print("T_cam: ", T_cam)

    # Ensure types are float64
    # T_cam = T_cam.astype(np.float64)
    T_mesh = T_mesh.astype(np.float64)

    # Calculate transformation matrix of mesh in camera frame
    # T_mesh_in_cam = np.linalg.inv(T_cam) @ T_mesh
    T_mesh_in_cam = T_mesh

    # Debug: Verify T_mesh_in_cam
    ##print("T_mesh_in_cam: ", T_mesh_in_cam)

    # Transform the object's vertices using T_mesh_in_cam
    transformed_vertices = object_vertices
    transformed_vertices_homogeneous = T_mesh_in_cam @ np.vstack(
        (transformed_vertices.T, np.ones(transformed_vertices.shape[0]))
    )
    transformed_vertices = transformed_vertices_homogeneous[:3, :].T

    # Debug: Check transformed object
    # ##print("Transformed vertices: ", transformed_vertices)

    # Update object mesh vertices
    object_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)

    # Extract new object translation in camera frame for further use if needed
    obj_trans_new_frame = T_mesh_in_cam[:3, 3]

    return object_mesh


def load_intrinsics_and_distortion(camera_id, base_path):
    calib_path = os.path.join(base_path, "calibs", "intrinsics", str(camera_id), "calibration.json")
    with open(calib_path, "r") as f:
        calib_data = json.load(f)
        color_intrinsics = calib_data["color"]
        return {
            "fx": color_intrinsics["fx"],
            "fy": color_intrinsics["fy"],
            "cx": color_intrinsics["cx"],
            "cy": color_intrinsics["cy"],
        }, {
            "k1": color_intrinsics["k1"],
            "k2": color_intrinsics["k2"],
            "k3": color_intrinsics["k3"],
            "p1": color_intrinsics["p1"],
            "p2": color_intrinsics["p2"],
        }


def project_to_image(point_3d, intrinsics, distortion_coeffs):
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    k1, k2, k3 = distortion_coeffs["k1"], distortion_coeffs["k2"], distortion_coeffs["k3"]
    p1, p2 = distortion_coeffs["p1"], distortion_coeffs["p2"]

    x_norm = point_3d[0] / point_3d[2]
    y_norm = point_3d[1] / point_3d[2]

    r2 = x_norm**2 + y_norm**2
    x_distorted = (
        x_norm * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) + 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
    )
    y_distorted = (
        y_norm * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) + 2 * p2 * x_norm * y_norm + p1 * (r2 + 2 * y_norm**2)
    )

    u = fx * x_distorted + cx
    v = fy * y_distorted + cy

    u = fx * x_norm + cx
    v = fy * y_norm + cy

    return int(u), int(v)


def axis_angle_loss(pred, true):
    # Assuming pred and true are [batch_size, 3] tensors where
    # the last dimension contains the [x, y, z] coordinates of the axis-angle vector

    # Normalize axis vectors
    pred_axis = F.normalize(pred, dim=1)
    true_axis = F.normalize(true, dim=1)

    # Calculate the cosine similarity between axes
    cos_sim = F.cosine_similarity(pred_axis, true_axis, dim=1)

    # Calculate angle magnitudes
    pred_angle = torch.norm(pred, dim=1)
    true_angle = torch.norm(true, dim=1)

    # Calculate circular distance for angles
    angle_diff_options = torch.stack(
        [
            torch.abs(pred_angle - true_angle),
            torch.abs(pred_angle - true_angle + 2 * np.pi),
            torch.abs(pred_angle - true_angle - 2 * np.pi),
        ],
        dim=-1,
    )

    angle_diff, _ = torch.min(angle_diff_options, dim=-1)

    # Combine the two losses
    # You can weight these terms as needed
    loss = 1 - cos_sim + angle_diff

    return torch.mean(loss)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.ReLU(),
            # nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class CombinedTrans(pl.LightningModule):
    def __init__(self, frames_subclip, masked_frames):
        super(CombinedTrans, self).__init__()

        self.save_hyperparameters()
        self.validation_losses = []
        self.frames_subclip = frames_subclip
        self.masked_frames = masked_frames
        self.automatic_optimization = False
        self.num_heads = 4
        self.d_model = 128
        self.mlp_output_pose = MLP(self.d_model, 3)
        self.mlp_output_trans = MLP(self.d_model, 3)
        self.mlp_smpl_pose = MLP(72, self.d_model)
        self.mlp_smpl_joints = MLP(72, self.d_model)
        self.mlp_obj_pose = MLP(3, self.d_model)
        self.mlp_obj_trans = MLP(3, self.d_model)
        self.best_avg_loss_val = float("inf")

        self.transformer_model_trans = nn.Transformer(
            d_model=self.d_model,
            nhead=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dropout=0.05,
            activation="gelu",
        )

        self.transformer_model_pose = nn.Transformer(
            d_model=self.d_model,
            nhead=self.num_heads,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dropout=0.05,
            activation="gelu",
        )

    def forward(self, smpl_pose, smpl_joints, obj_pose, obj_trans):
        # smpl_pose, smpl_joints, obj_pose, obj_trans = cam_data[-2][:]

        smpl_joints = smpl_joints.reshape(-1, self.frames_subclip, 72)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # print("SMPL Pose:", smpl_pose.shape)
        # print("SMPL Joints:", smpl_joints.shape)
        # print("Object Pose:", obj_pose.shape)
        # print("Object Trans:", obj_trans.shape)

        masked_obj_pose = obj_pose.clone()
        masked_obj_trans = obj_trans.clone()

        masked_obj_pose[:, -self.masked_frames :, :] = 0
        masked_obj_trans[:, -self.masked_frames :, :] = 0

        # Move each tensor to the specified device
        smpl_pose = smpl_pose.to(device)
        smpl_joints = smpl_joints.to(device)
        masked_obj_pose = masked_obj_pose.to(device)
        masked_obj_trans = masked_obj_trans.to(device)
        obj_pose = obj_pose.to(device)
        obj_trans = obj_trans.to(device)

        # Embedding inputs
        embedded_smpl_pose = self.mlp_smpl_pose(smpl_pose)
        embedded_obj_pose = self.mlp_obj_pose(masked_obj_pose)
        embedded_smpl_joints = self.mlp_smpl_joints(smpl_joints)
        embedded_obj_trans = self.mlp_obj_trans(masked_obj_trans)

        # #Print shapes of the embedded tensors
        # print("Embedded SMPL Pose Shape:", embedded_smpl_pose.shape)
        # print("Embedded Object Pose Shape:", embedded_obj_pose.shape)
        # print("Embedded SMPL Joints Shape:", embedded_smpl_joints.shape)
        # print("Embedded Object Trans Shape:", embedded_obj_trans.shape)

        # Initialize tgt_mask
        # tgt_mask = torch.zeros(wandb.config.batch_size * self.num_heads, self.frames_subclip, self.d_model, self.d_model, dtype=torch.bool)
        tgt_mask = torch.zeros(
            wandb.config.batch_size * self.num_heads, self.frames_subclip, self.frames_subclip, dtype=torch.bool
        ).to(device)
        # tgt_mask = torch.zeros(48, self.frames_subclip, self.frames_subclip, dtype=torch.bool).to(device)

        # Iterate to set the last self.masked_frames rows of the upper diagonal matrix to True
        for i in range(wandb.config.batch_size * self.num_heads):
            for row in range(self.frames_subclip - self.masked_frames, self.frames_subclip):
                tgt_mask[i, row, row:] = True  # Set the elements on and above the diagonal to True

        # Separe pose and joints
        # Transformer models

        predicted_obj_pose_emb = self.transformer_model_pose(
            embedded_smpl_pose.permute(1, 0, 2), embedded_obj_pose.permute(1, 0, 2), tgt_mask=None
        )  # tgt_mask)
        predicted_obj_trans_emb = self.transformer_model_trans(
            embedded_smpl_joints.permute(1, 0, 2), embedded_obj_trans.permute(1, 0, 2), tgt_mask=None
        )  # tgt_mask)

        predicted_obj_pose = self.mlp_output_pose(predicted_obj_pose_emb.permute(1, 0, 2))
        predicted_obj_trans = self.mlp_output_trans(predicted_obj_trans_emb.permute(1, 0, 2))

        # #Print dimensions of the tensors
        # #print("Dimensions of Predicted Object Pose:", predicted_obj_pose.shape)
        # #print("Dimensions of Predicted Object Trans:", predicted_obj_trans.shape)

        return predicted_obj_pose, predicted_obj_trans


####################
def main():
    # Dataset creation
    # dataset = []

    # Set scene
    # identifiers = ["Date03_Sub03_tablesmall_lift", "Date03_Sub03_tablesquare_move" ,"Date03_Sub03_stool_sit","Date03_Sub03_stool_lift", "Date03_Sub03_plasticcontainer", "Date03_Sub03_chairwood_sit", "Date03_Sub03_boxmedium", "Date03_Sub03_boxlarge",\
    # "Date03_Sub05_tablesquare", "Date03_Sub05_suitcase", "Date03_Sub05_stool", "Date03_Sub05_boxmedium", "Date03_Sub04_tablesquare_sit", "Date03_Sub04_suitcase_lift", "Date03_Sub04_plasticcontainer_lift", "Date03_Sub04_boxlong", "Date03_Sub03_yogamat"]
    identifiers = [
        "Date03_Sub03_boxmedium",
        "Date03_Sub03_stool_sit",
        "Date03_Sub03_stool_lift",
        "Date03_Sub03_plasticcontainer",
        "Date03_Sub03_chairwood_sit",
        "Date03_Sub03_boxlarge",
        "Date03_Sub05_tablesquare",
        "Date03_Sub05_suitcase",
        "Date03_Sub05_stool",
        "Date03_Sub04_tablesquare_sit",
        "Date03_Sub04_suitcase_lift",
        "Date03_Sub04_boxlong",
        "Date03_Sub03_yogamat",
    ]

    for identifier in identifiers:
        print("Processing:", identifier, flush=True)

        # Change .pt name when creating a new one
        data_file_path = os.path.join(
            "/srv/beegfs02/scratch/3dhumanobjint/data/H2O/datasets/30fps_numpy", identifier + ".pkl"
        )

        temp_dir = "/scratch_net/biwidl307/lgermano/crossvit/visualizations/temp_frames"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            ##print(f"Created directory: {temp_dir}")
        else:
            # Delete all files and subdirectories in the directory
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    return f"Failed to delete {file_path}. Reason: {e}"

            ##print(f"Directory {temp_dir} already exists. Its contents have been deleted.")

        # Check if the data has already been saved
        if os.path.exists(data_file_path):
            # Load the saved data
            with open(data_file_path, "rb") as f:
                cam_data = pickle.load(f)

        ###print(cam_data[0][0].keys())
        # for cam_id in range(4):
        #     for idx in range(20):
        #         ##print(dataset[cam_id][idx].values())
        frames_subclip = 12  # 115/12 = 9
        masked_frames = 4
        model = CombinedTrans(frames_subclip, masked_frames)

        # Specify device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to device
        model.to(device)
        model_path = f"/srv/beegfs02/scratch/3dhumanobjint/data/H2O/trained_models/model_ethereal-frost-2985cross_att_12_4_zeros_epoch_9.pt"
        # model_path = f"/srv/beegfs02/scratch/3dhumanobjint/data/H2O/trained_models/model_robust-smoke-3015cross_att_12_4_axis_angle_loss_from_checkpoint_pose_only_epoch_3.pt"
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)

        # Set the model to evaluation mode
        model.eval()

        #######################################################################s

        # def compute_ADD(GT_vertices, candidate_vertices):
        #     # Convert the data to PyTorch tensors and transfer to GPU
        #     GT_vertices_torch = torch.tensor(GT_vertices, device='cuda')
        #     candidate_vertices_torch = torch.tensor(candidate_vertices, device='cuda')

        #     # Compute the distances
        #     distances = torch.norm(GT_vertices_torch - candidate_vertices_torch, dim=1)
        #     return torch.mean(distances).item()  # Convert the result back to a Python scalar

        # def compute_ADD_S(GT_vertices, candidate_vertices):
        #     # Convert the data to PyTorch tensors and transfer to GPU
        #     GT_vertices_torch = torch.tensor(GT_vertices, device='cuda')
        #     candidate_vertices_torch = torch.tensor(candidate_vertices, device='cuda')

        #     # Compute the pairwise distances
        #     distances = torch.cdist(GT_vertices_torch.unsqueeze(0), candidate_vertices_torch.unsqueeze(0)).squeeze(0)

        #     # Find the minimum distance for each point in GT_vertices
        #     min_distances = torch.min(distances, dim=1).values

        #    return torch.mean(min_distances).item()  # Convert the result back to a Python scalar

        def compute_cd(GT_vertices, candidate_vertices):
            # Convert lists to numpy arrays for efficient computation
            # For each point in GT consider the closest point in PR
            # And viceversa. Sum the averages.
            kdtree1 = cKDTree(GT_vertices)
            dists1, indices1 = kdtree1.query(candidate_vertices)
            kdtree2 = cKDTree(candidate_vertices)
            dists2, indices2 = kdtree2.query(GT_vertices)
            return 0.5 * (
                dists1.mean() + dists2.mean()
            )  #!NOTE should not be mean of all, see https://pdal.io/en/stable/apps/chamfer.html

        def add_err(pred, gt):
            """
            Average Distance of Model Points for objects with no indistinguishable views
            - by Hinterstoisser et al. (ACCV 2012).
            The direct distance is considered
            """
            #   pred_pts = (pred@to_homo(model_pts).T).T[:,:3]
            #   gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
            e = np.linalg.norm(pred - gt, axis=1).mean()
            return e

        def adi_err(pred_pts, gt_pts):
            """
            @pred: 4x4 mat
            @gt:
            @model: (N,3)
            For each GT point, the distnace to the closest PR point is considered
            """
            # = (pred@to_homo(model_pts).T).T[:,:3]
            # gt_pts = (gt@to_homo(model_pts).T).T[:,:3]
            nn_index = cKDTree(pred_pts)
            nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
            e = nn_dists.mean()
            return e

        def compute_auc(rec, max_val=0.1):
            if len(rec) == 0:
                return 0
            rec = np.sort(np.array(rec))
            n = len(rec)
            ##print(n)
            prec = np.arange(1, n + 1) / float(n)
            rec = rec.reshape(-1)
            prec = prec.reshape(-1)
            index = np.where(rec < max_val)[0]
            rec = rec[index]
            prec = prec[index]

            if len(prec) == 0:
                return 0

            mrec = [0, *list(rec), max_val]
            # Only add prec[-1] if prec is not empty
            mpre = [0, *list(prec)] + ([prec[-1]] if len(prec) > 0 else [])

            for i in range(1, len(mpre)):
                mpre[i] = max(mpre[i], mpre[i - 1])
            mpre = np.array(mpre)
            mrec = np.array(mrec)
            i = np.where(mrec[1:] != mrec[: len(mrec) - 1])[0] + 1
            ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) / max_val
            return ap

        # Initialize lists to store ADD and ADD-S values for AUC computation
        # cam_data = {0: [], 1: [], 2: [], 3: []}  # Initialize a dictionary to hold data for each camera

        all_ADD_values = {0: [], 1: [], 2: [], 3: []}
        all_ADD_S_values = {0: [], 1: [], 2: [], 3: []}
        all_CD_values = {0: [], 1: [], 2: [], 3: []}

        max_th = 0.10

        ##########################################################################

        # Initialize variables to store the previous object pose and translation for each camera
        frame_idx = 0
        prev_obj_pose = None
        prev_obj_trans = None
        items = [None] * 4
        # Process interpolated frames
        # for idx in range(0,len(cam_data[2]),masked_frames):
        # make a masked_frames step, only if prev_ variables are set to None
        # for idx in range(0,len(cam_data[2]) - frames_subclip +1, frames_subclip):
        last_frame = min(len(cam_data[2]) - frames_subclip + 1, 100)
        for idx in range(0, last_frame, 1):
            images = []
            for cam_id in [2]:
                # len = 40 (inx 0 -39,), mask = 4 (index of first masked = 36)
                obj_pose = cam_data[cam_id][idx + frames_subclip - masked_frames]["obj_pose"]
                obj_trans = cam_data[cam_id][idx + frames_subclip - masked_frames]["obj_trans"]
                identifier = cam_data[cam_id][idx]["scene"]
                betas = cam_data[cam_id][idx + frames_subclip - masked_frames]["betas"]
                smpl_pose = cam_data[cam_id][idx + frames_subclip - masked_frames]["pose"]
                smpl_trans = cam_data[cam_id][idx + frames_subclip - masked_frames]["trans"]
                smpl_joints = cam_data[cam_id][idx + frames_subclip - masked_frames]["joints"]
                date = cam_data[cam_id][idx]["date"]
                obj_template_path = cam_data[cam_id][idx]["obj_template_path"]

                # There is no image path for the momesmpl_posent...
                img = np.ones((1800, 1800, 4), dtype=np.uint8) * 255

                base_path = "/scratch_net/biwidl307_second/lgermano/behave"

                cam_params = load_config(cam_id, base_path, date)
                intrinsics_cam, distortion_cam = load_intrinsics_and_distortion(cam_id, base_path)

                # Rendering
                selected_joints, verts, projected_verts, smpl_faces = render_smpl(
                    smpl_pose, smpl_trans, betas, intrinsics_cam, distortion_cam, img
                )
                projected_selected_joints = [
                    project_to_image(joint, intrinsics_cam, distortion_cam) for joint in selected_joints
                ]

                # Specify device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                scene = cam_data[cam_id][0]["scene"]
                start_idx = idx
                end_idx = idx + frames_subclip
                subclip_data = cam_data[cam_id][start_idx:end_idx]

                if prev_obj_pose is not None and prev_obj_trans is not None:
                    items[0] = torch.tensor(
                        np.vstack([subclip_data[i]["pose"] for i in range(len(subclip_data))]), dtype=torch.float32
                    )
                    items[1] = torch.tensor(
                        np.vstack([subclip_data[i]["joints"] for i in range(len(subclip_data))]), dtype=torch.float32
                    )

                    items[2] = torch.roll(items[2], -1, 0)
                    items[2][-masked_frames - 1] = prev_obj_pose

                    items[3] = torch.roll(items[3], -1, 0)
                    items[3][-masked_frames - 1] = prev_obj_trans

                else:
                    keys = ["pose", "joints", "obj_pose", "obj_trans"]
                    items = [
                        torch.tensor(
                            np.vstack([subclip_data[i][key] for i in range(len(subclip_data))]), dtype=torch.float32
                        )
                        for key in keys
                    ]

                candidate_obj_pose_tensor, candidate_obj_trans_tensor = model(
                    items[0].unsqueeze(0).to(device),
                    items[1].unsqueeze(0).to(device),
                    items[2].unsqueeze(0).to(device),
                    items[3].unsqueeze(0).to(device),
                )

                # Store the candidate pose and translation for use in the next iteration
                # prev_obj_pose = None #torch.from_numpy(obj_pose) #candidate_obj_pose_tensor[0,-masked_frames,:]
                prev_obj_pose = candidate_obj_pose_tensor[0, -masked_frames, :]
                # prev_obj_trans = None #torch.from_numpy(obj_trans) #candidate_obj_trans_tensor[0,-masked_frames,:]
                prev_obj_trans = candidate_obj_trans_tensor[0, -masked_frames, :]

                candidate_obj_pose = candidate_obj_pose_tensor.cpu().detach().numpy()
                candidate_obj_trans = candidate_obj_trans_tensor.cpu().detach().numpy()

                # Plot from the first predicted frame onward
                transformed_object = plot_obj_in_camera_frame(
                    candidate_obj_pose[0, -masked_frames, :],
                    candidate_obj_trans[0, -masked_frames, :],
                    obj_template_path,
                )
                vertices_np = np.asarray(transformed_object.vertices)  # Convert directly to numpy array
                obj_projected_verts = [project_to_image(vert, intrinsics_cam, distortion_cam) for vert in vertices_np]
                faces_np = np.asarray(transformed_object.triangles)  # Convert the triangles to numpy array

                # GT
                GT_obj = plot_obj_in_camera_frame(obj_pose, obj_trans, obj_template_path)
                candidate_vertices_np = np.asarray(GT_obj.vertices)  # Convert directly to numpy array
                candidate_obj_projected_verts = [
                    project_to_image(vert, intrinsics_cam, distortion_cam) for vert in candidate_vertices_np
                ]
                candidate_faces_np = np.asarray(GT_obj.triangles)  # Convert the triangles to numpy array

                img = project_mesh_on_image(
                    img,
                    projected_verts,
                    smpl_faces,
                    obj_projected_verts,
                    faces_np,
                    candidate_obj_projected_verts,
                    candidate_faces_np,
                    projected_selected_joints,
                )

                # candidate_obj_pose = candidate_obj_pose[0,-masked_frames,:]
                # candidate_obj_trans = candidate_obj_trans[0,-masked_frames,:]

                # Calculate MSE for obj_pose
                error_pose = candidate_obj_pose - obj_pose
                obj_pose_loss = np.mean(np.square(error_pose))

                # Calculate MSE for obj_trans
                error_trans = candidate_obj_trans - obj_trans
                obj_trans_loss = np.mean(np.square(error_trans))

                # Add cam_id onto the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                fontColor = (255, 255, 255)  # White color
                lineType = 4

                # Text positions
                bottomLeftCornerOfText = (10, 50)  # Position for Camera ID
                bottomRightCornerOfText1 = (10, 100)  # Position for MSE pose
                bottomRightCornerOfText2 = (10, 150)  # Position for MSE trans

                # Put text on the image
                cv2.putText(img, f"Camera ID: {cam_id}", bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

                cv2.putText(
                    img,
                    f"MSE pose: {obj_pose_loss:.4f}",
                    bottomRightCornerOfText1,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )

                cv2.putText(
                    img,
                    f"MSE trans: {obj_trans_loss:.4f}",
                    bottomRightCornerOfText2,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )

                ##print(f"Rendered SMPL on image for camera {cam_id}.")
                images.append(img)

                ##########################################################################
                # Simplify the meshes to 100 points
                final_num_nodes = 100

                GT_obj_sim = o3d.geometry.PointCloud()
                new_positions, _ = simplify_mesh(
                    np.asarray(GT_obj.vertices), np.asarray(GT_obj.triangles).astype(np.uint32), final_num_nodes
                )
                GT_obj_sim.points = o3d.utility.Vector3dVector(np.asarray(new_positions))

                transformed_obj_sim = o3d.geometry.PointCloud()
                # new_positions2, _ = simplify_mesh(np.asarray(transformed_object.vertices), np.asarray(transformed_object.triangles).astype(np.uint32), final_num_nodes)

                # Test
                new_positions2, _ = simplify_mesh(
                    np.asarray(GT_obj.vertices), np.asarray(GT_obj.triangles).astype(np.uint32), final_num_nodes
                )
                transformed_obj_sim.points = o3d.utility.Vector3dVector(np.asarray(new_positions2))

                # # Convert the meshes to point clouds by using their vertices
                # GT_obj_pcd = o3d.geometry.PointCloud()
                # GT_obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(GT_obj.vertices))

                # candidate_obj_pcd = o3d.geometry.PointCloud()
                # candidate_obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(transformed_object.vertices))
                # #candidate_obj_pcd.points = o3d.utility.Vector3dVector(np.asarray(GT_obj.vertices))

                # Convert Open3D point clouds to numpy arrays
                GT_vertices = np.asarray(GT_obj_sim.points)
                candidate_vertices = np.asarray(transformed_obj_sim.points)

                # # ###print the lengths of the numpy arrays
                # ###print("Length of GT_obj_np:", len(GT_obj_np))
                # ###print("Length of candidate_obj_np:", len(candidate_obj_np))

                # # Calculate 5% of the total number of points
                # num_points = GT_obj_np.shape[0]
                # num_sampled_points = 100 #int(num_points * 0.0005)

                # # Generate random indices
                # np.random.seed(0) # for reproducibility
                # random_indices = np.random.choice(num_points, num_sampled_points, replace=False)

                # #Select the points using these indices
                # GT_vertices = GT_obj_np[random_indices]
                # candidate_vertices = candidate_obj_np[random_indices]

                # #Select the points using these indices
                # GT_vertices = GT_obj_np#[random_indices]
                # candidate_vertices = candidate_obj_np#[random_indices]

                # Now you can compute ADD and ADD-S and CD
                add = add_err(candidate_vertices, GT_vertices)
                add_s = adi_err(candidate_vertices, GT_vertices)
                cd = compute_cd(GT_vertices, candidate_vertices)

                ###print the computed values
                ##print("ADD:", add)
                ##print("ADD-S:", add_s)
                ##print("CD:", cd)

                # Store ADD and ADD-S values
                # scene_cam_ADD_values.append(add)
                # scene_cam_ADD_S_values.append(add_s)
                # scene_cam_CD_values.append(cd)
                all_ADD_values[cam_id].append(add)
                all_ADD_S_values[cam_id].append(add_s)
                all_CD_values[cam_id].append(cd)

                ############################################################################

            ##print("\nCombining images...")
            num_images = len(images)

            # Handle cases for different number of images
            if num_images == 1:
                all_images = images[0]
            elif num_images == 2:
                all_images = np.hstack((images[0], images[1]))
            elif num_images == 3:
                top_row = np.hstack((images[0], images[1]))
                all_images = np.vstack((top_row, images[2]))
            elif num_images == 4:
                top_row = np.hstack((images[0], images[1]))
                bottom_row = np.hstack((images[2], images[3]))
                all_images = np.vstack((top_row, bottom_row))

            height, width = all_images.shape[:2]
            new_width = int(width * 1.0)
            new_height = int(height * 1.0)
            new_width = new_width if new_width % 2 == 0 else new_width + 1
            new_height = new_height if new_height % 2 == 0 else new_height + 1

            resized_image = cv2.resize(all_images, (new_width, new_height))
            cv2.putText(
                resized_image, identifier, (10, new_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4
            )
            frame_path = os.path.join(temp_dir, "frame_{:04d}.png".format(frame_idx))
            success = cv2.imwrite(frame_path, resized_image)
            # print(f"Saving frame at {frame_path}. Success: {success}")

            frame_idx += 1

        # print("\nCreating video...")

        ########################################################

        for cam_id in [2]:
            auc_ADD = compute_auc(np.array(all_ADD_values[cam_id]), max_th) * 100
            auc_ADD_S = compute_auc(np.array(all_ADD_S_values[cam_id]), max_th) * 100
            cd_mean = sum(all_CD_values[cam_id]) / len(all_CD_values[cam_id])

            print(
                f"AUC for scene {identifier}, camera {cam_id} - ADD: {auc_ADD:.2f}%, ADD-S: {auc_ADD_S:.2f}%, CD [m]: {cd_mean:.2f}",
                flush=True,
            )

        #######################################################

        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        base_video_path = os.path.join("/scratch_net/biwidl307/lgermano/crossvit/visualizations/", identifier)
        # base_video_path = os.path.join("/srv/beegfs02/scratch/3dhumanobjint/data/visualizations/", identifier)

        video_path = f"{base_video_path}_{timestamp}.mp4"

        subprocess.call(
            [
                "ffmpeg",
                "-r",
                "2",
                "-i",
                os.path.join(temp_dir, "frame_%04d.png"),
                "-vcodec",
                "libx264",
                "-crf",
                "30",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
        )
        print(f"Video saved at {video_path}.")

        # print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        cv2.destroyAllWindows()
        # print("Cleanup complete.")


if __name__ == "__main__":
    main()
