
import numpy as np
import os 
import json
import torch
import cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp



mano_ncomps=45
mano_root= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mano_models')
obj_path={'left':os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mano_models',"MANO_UV_left.obj"),
            'right':os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mano_models',"MANO_UV_right.obj")}
participants=['participant_1','participant_2','participant_3','participant_4','participant_5','participant_6','participant_7','participant_8','participant_9','participant_10','participant_11','participant_12','participant_13','participant_14','participant_15','participant_16','participant_17','participant_18','participant_19','participant_20','participant_21','participant_22','participant_23','participant_24','participant_25','participant_26']
cameras={0:"camera_d",1:"camera_1",2:"camera_2",3:"camera_3",4:"camera_4",5:"camera_5",6:"camera_6",7:"camera_7"}
colormaps = {
            'GRAY': cv2.COLORMAP_BONE,  # Using BONE as it closely resembles GRAY
            'INFERNO': cv2.COLORMAP_INFERNO,
            'OCEAN': cv2.COLORMAP_OCEAN,
            'JET': cv2.COLORMAP_JET,
            'HOT': cv2.COLORMAP_HOT,

        }
sequences = [
    "draw_word_5x_right", "calibration_routine_left", "calibration_routine_right",
    "grasp-edge_curled_thumb-down_5x_left", "grasp-edge_uncurled_thumb-down_5x_left",
    "grasp-edge_curled_thumb-down_5x_right", "grasp-edge_curled_thumb-up_5x_right",
    "grasp-edge_uncurled_thumb-down_5x_right", "index_press_high_x5_left",
    "index_press_low_x5_left", "index_press_no-contact_x5_left", "index_press_high_x5_right",
    "index_press_low_x5_right", "index_press_no-contact_x5_right", "index_press_pull_x5_left",
    "index_press_push_x5_left", "index_press_pull_x5_right", "index_press_push_x5_right",
    "index_press_rotate-left_x5_left", "index_press_rotate-right_x5_left",
    "index_press_rotate-left_x5_right", "index_press_rotate-right_x5_right",
    "pinch-zoom_5x_left", "pinch_thumb-down_high_5x_left", "pinch_thumb-down_low_5x_left",
    "pinch_thumb-down_no-contact_5x_left", "pinch_thumb-down_high_5x_right",
    "pinch_thumb-down_low_5x_right", "pinch_thumb-down_no-contact_5x_right",
    "press_cupped_onebyone_high_3x_left", "press_cupped_onebyone_low_3x_left",
    "press_cupped_onebyone_high_3x_right", "press_cupped_onebyone_low_3x_right",
    "press_fingers_high_5x_left", "press_fingers_low_5x_left",
    "press_fingers_no-contact_5x_left", "press_fingers_low_5x_right",
    "press_fingers_no-contact_5x_right", "press_flat_onebyone_high_3x_left",
    "press_flat_onebyone_low_3x_left", "press_flat_onebyone_high_3x_right",
    "press_flat_onebyone_low_3x_right", "press_palm-and-fingers_high_x5_left",
    "press_palm-and-fingers_low_x5_left", "press_palm-and-fingers_no-contact_x5_left",
    "press_palm_high_x5_left", "press_palm_low_x5_left", "press_palm_low_x5_right",
    "press_palm_no-contact_x5_right", "pull-towards_5x_left", "push-away_5x_left",
    "pull-towards_5x_right", "draw_word_5x_left", "press_palm-and-fingers_high_x5_right",
    "press_palm-and-fingers_low_x5_right", "press_palm-and-fingers_no-contact_x5_right",
    "press_palm_no-contact_x5_left", "press_palm_high_x5_right", "push-away_5x_right",
    "type_ipad_5x_left", "type_ipad_5x_right", "grasp-edge_curled_thumb-up_5x_left",
    "pinch-zoom_5x_right", "press_fingers_high_5x_right"
]

def get_camera_parameters(camera_calibration,mm2m=True):
    ImageSizeX, ImageSizeY = camera_calibration["ImageSizeX"], camera_calibration["ImageSizeY"]
    fx, fy = camera_calibration["fx"], camera_calibration["fy"]
    cx, cy = camera_calibration["cx"], camera_calibration["cy"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    # Extract distortion coefficients
    #dist_coeffs = torch.tensor([camera_calibration["k1"], camera_calibration["k2"], camera_calibration["p1"], camera_calibration["p2"], camera_calibration["k3"]])
    dist_coeffs=torch.tensor([])
    # Extract extrinsic parameters
    if "ModelViewMatrix" in camera_calibration:
        R= torch.tensor(camera_calibration["ModelViewMatrix"],dtype=torch.float32)[:3, :3]
        T = torch.tensor(camera_calibration["ModelViewMatrix"],dtype=torch.float32)[:3, 3].reshape(3, 1)
        if mm2m:
            T=T/1000
    else:
        R = torch.eye(3, dtype=torch.float32)
        T = torch.zeros(3, 1, dtype=torch.float32)
    return R.numpy(),T.numpy(),K.numpy(),dist_coeffs.numpy(), [ImageSizeX, ImageSizeY]

def read_camera_json(file):
    
    with open(file) as f:
        data = json.load(f)
    camera_calibrations=data['camera_calibrations']
    camera_parameters={}
    for camera_index in camera_calibrations.keys():
        camera_calibration=camera_calibrations[camera_index]
        R,T,K,dist_coeffs,image_size=get_camera_parameters(camera_calibration)
        
        camera_parameters[int(camera_index)]={'R':R,'T':T,'K':K,'dist_coeffs':dist_coeffs,'image_size':image_size}
        
    return camera_parameters

def read_pressure_bin(file,height=105,width=185):
    with open(file, 'rb') as f:
        data = f.read()
    data = np.frombuffer(data, dtype=np.float32)
    data = data.reshape(height, width)
    return data

def which_side(sequence):
    if "right" in sequence[-6:]:
        hand_side="right"
    elif "left" in sequence[-5:]:
        hand_side="left"
    else:
        print("Invalid folder name")
        return None
    return hand_side

        
def decompose_and_interpolate(camera_poses):
    poses = {frame: {"R": np.array(data)[:3, :3], "T": np.array(data)[:3, 3],"frame":frame}
                for frame, data in camera_poses.items()}

    frame_numbers = sorted(camera_poses.keys(), key=lambda x: int(x))
    complete_range = [f"{i:06d}" for i in range(1, int(frame_numbers[-1]) + 1)]
    missing_frames = set(complete_range) - set(frame_numbers)

    for missing_frame in missing_frames:
        previous_frames = [frame for frame in frame_numbers if frame < missing_frame]
        next_frames = [frame for frame in frame_numbers if frame > missing_frame]
        if previous_frames and next_frames:
            prev_frame = max(previous_frames)
            next_frame = min(next_frames)
            total_frames = int(next_frame) - int(prev_frame)
            frame_offset = int(missing_frame) - int(prev_frame)
            fraction = frame_offset / total_frames

            prev_pose = np.array(camera_poses[prev_frame])
            next_pose = np.array(camera_poses[next_frame])
            
            prev_quat = Rot.from_matrix(prev_pose[:3, :3])
            next_quat = Rot.from_matrix(next_pose[:3, :3])
            slerp = Slerp([0,1],Rot.from_quat([prev_quat.as_quat(), next_quat.as_quat()]))
            slerp_result=slerp([fraction])[0]
            interpolated_rotation = slerp_result.as_matrix()
            
            prev_trans = prev_pose[:3, 3]
            next_trans = next_pose[:3, 3]
            interpolated_translation = (1 - fraction) * prev_trans + fraction * next_trans
            
            poses[missing_frame] = {"R": interpolated_rotation, "T": interpolated_translation,"frame":missing_frame}
    return poses



def parse_obj_for_uv_mapping(file_path):
    texcoords = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('vt '):
                parts = line.split()
                texcoords.append((float(parts[1]), float(parts[2])))
            elif line.startswith('f '):
                parts = line.split()
                face = []
                for part in parts[1:]:
                    components = part.split('/')
                    if len(components) > 1:
                        face.append(int(components[1]) - 1)  # Adjust for 0-based index
                if face:
                    faces.append(face)
    return texcoords, faces

def draw_uv_map_with_opencv(texcoords, faces, image_size=500):
    # Create a white image
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
    for face in faces:
        pts = np.array([[int(texcoords[idx][0] * image_size), int((1 - texcoords[idx][1]) * image_size)] for idx in face], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(125, 125, 125), thickness=1)
    return img