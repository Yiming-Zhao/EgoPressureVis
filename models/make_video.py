import cv2
import math 
import numpy as np
import os
import torch
import time
import argparse
import json
import pickle
from models.sensel_projection import project_rectangle, sensel_corners_3D
from models.visualization import render_obj_foreground, get_2d_points, get_3d_joints_in_camera_frame,draw_skeleton,ManoObject,apply_colormap_on_depth_image
from models.pressure_util import  get_force_overlay_img,pressure_to_colormap
from models.io import read_pressure_bin,which_side,decompose_and_interpolate,read_camera_json,parse_obj_for_uv_mapping,draw_uv_map_with_opencv
from models.io import participants as known_participants
from models.io import sequences as known_sequences
from models.io import cameras,colormaps,obj_path




class video_config:
    
    def __init__(self):
        self.fps = 30
        self.save_images = False
        self.visible_camera_views = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False} 
        self.visualization_config = {
            "depth_colormap": "JET",
            "force_color_map": "INFERNO",
            "mesh_color": [0.95, 0.95, 0.95],
            "mesh_alpha": 0.85,
            "joint_thickness": 2
        }
        self.visibility = {
            "RGB": True,
            "depth": False,
            "mesh": False,
            "vertices_disp": False,
            "wrapped_force": False,
            "sensel_area": False,
            "joints2D": False,
            "uv_pressure": False
        }
        self.frame_range = [0, -1]
        self.single_view_resolution = [640, 360]
        self.save_images_path = "./video_frames"
        self.save_video_path = "./videos"
        self.video_postfix_time = False
     
        self.__hints__={"fps":"(int:30) frames per second",
                        "save_images":"(bool:False) to save images of each frame",
                        "visible_camera_views":"(list[int]: 0 1) List of camera views to be visualized, 0(EgoCam),1,2,3,4,5,6,7(StaticCam),-1(no camera)",
                        "visualization_config":{"depth_colormap":"(str:'GRAY') Colormap for depth visualization,'GRAY','INFERNO','OCEAN','JET','HOT'",
                        "force_color_map":"(str:'OCEAN') Colormap for force visualization,'GRAY','INFERNO','OCEAN','HOT'",
                        "mesh_color":"(list[float][3]: 0.95 0.95 0.95) Color of the mesh, RGB value with range [0,1]",
                        "mesh_alpha":"(float:0.85) Transparency of the mesh of range [0,1]",
                        "joint_thickness":"(int:4) Thickness of the drawn 2d joints and skelton"},
                        "visibility":{ "RGB":"(bool:True) Visualize RGB images",
                        "depth":"(bool:False) Visualize depth images, ignored if RGB is True",
                        "mesh":"(bool:False) Visualize mesh",
                        "vertices_disp":"(bool:False) apply vertices displacement",
                        "wrapped_force":"(bool:False) Visualize wrapped force map on image",
                        "sensel_area":"(bool:False) Visualize sensel touch pad area",
                        "joints2D":"(bool:False) Visualize 2D joints and skelton",
                        "uv_pressure":"(bool:False) Visualize pressure map on UV map"
                        },
                        "frame_range":"(list[int][2]: 0 -1) Range of frames to be visualized, -1 till the last frame of seq ",
                        "single_view_resolution":"(list[int][2]: 640 360) Resolution of single view",
                        "save_images_path":"(str:'./video_frames')Path to save images of each frame",
                        "save_video_path":"(str:'./videos') Path to save video",
                        "video_postfix_time":"(bool:False) Add time postfix to video name"}
        self.__app_hints__ = {
            "fps": "Frames per second",
            "save_images": "to save images of each frame",
            "visible_camera_views": "List of camera views to be visualized, 0(EgoCam),1,2,3,4,5,6,7(StaticCam),-1(no camera)",
            "visualization_config": {
                "depth_colormap": "Colormap for depth visualization,'GRAY','INFERNO','OCEAN','JET','HOT'",
                "force_color_map": "Colormap for force visualization,'GRAY','INFERNO','OCEAN','HOT'",
                "mesh_color": "Color of the mesh, RGB value with range [0,1]",
                "mesh_alpha": "Transparency of the mesh of range [0,1]",
                "joint_thickness": "Thickness of the drawn 2d joints and skelton"
            },
            "visibility": {
                "RGB": " Visualize RGB images",
                "depth": "Visualize depth images",
                "mesh": "Visualize mesh",
                "vertices_disp": "Apply vertices displacement",
                "wrapped_force": "Visualize wrapped force map on image",
                "sensel_area": "Visualize sensel touch pad area",
                "joints2D": "Visualize 2D joints and skelton",
                "uv_pressure": "Visualize pressure map on UV map"
            },
            "frame_range": "Range of frames to be visualized, -1 till the last frame of the sequence",
            "single_view_resolution": "Resolution of single view",
            "save_images_path": "Path to save images of each frame",
            "save_video_path": "Path to save video",
            "video_postfix_time": "Add time postfix to video name"
        }
   
    def Set(self,attr,value):
        if hasattr(self,attr):           
            setattr(self,attr,value)#
            return True
        else:
            return False
    def Get(self,attr):
        if hasattr(self,attr):
            return getattr(self,attr)
        else:
            return None
    def display(self):
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self,attr)):	
                print(f"{attr}={getattr(self,attr)}")
def print_progress(message):
    if not message['newline']:
        print(f"\r{message['message']}", end='')
    else:
        print(f"\n{message['message']}")

class Subplot:
    def __init__(self, resolution, views_count):
        self.single_view_resolution = resolution
        self.views_count = views_count
        self.rows, self.cols = self.calculate_layout()
        self.resolution=(self.cols * self.single_view_resolution[0],self.rows * self.single_view_resolution[1])
    def calculate_layout(self):
        # Calculate the total number of pixels for one view
        view_width, view_height = self.single_view_resolution
        
        # Start with a square-like layout
        rows = math.floor(math.sqrt(self.views_count))
        max_edge=math.ceil(math.sqrt(self.views_count))
        cols = math.ceil(self.views_count / rows)
        
        # Adjust rows and columns to ensure the layout fits the resolution
        while rows * cols < self.views_count:
            if (cols + 1) * view_width <= (rows + 1) * view_height:
                cols += 1
            else:
                rows += 1
        
        # Make sure the layout is as close to a square as possible
        possible_layouts = []
        for r in range(1, self.views_count + 1):
            c = math.ceil(self.views_count / r)
            if r<=max_edge and c<=max_edge:
                possible_layouts.append((r, c))
        # Select the layout that gives the smallest difference between width and height
        min_diff = float('inf')
        best_layout = (rows, cols)
        for r, c in possible_layouts:
            width = c * view_width
            height = r * view_height
            diff = abs(width - height)
            if diff < min_diff:
                min_diff = diff
                best_layout = (r, c)
        
        return best_layout



    def resize_and_center_image(self,img,texts):
        view_width, view_height = self.single_view_resolution
        
        # Read the image

        
        img_height, img_width = img.shape[:2]
        
        # Calculate aspect ratios
        img_aspect = img_width / img_height
        view_aspect = view_width / view_height
        
        # Resize image while keeping aspect ratio
        if img_aspect > view_aspect:
            new_width = view_width
            new_height = int(new_width / img_aspect)
        else:
            new_height = view_height
            new_width = int(new_height * img_aspect)
        
        resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_image = resized_image.astype(np.uint8)
        # Create new image with single view resolution and center the resized image
        new_image = np.full((view_height, view_width, 3), 255, dtype=np.uint8)

        offset_x = (view_width - new_width) // 2
        offset_y = (view_height - new_height) // 2
        new_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_image
        if 'color' not in texts: 
            text_color=(255,255,255)
        else:
            text_color=texts['color']
        text=texts['text']  
        cv2.putText(new_image,text,(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,text_color,1)	
        return new_image

    def create_total_view(self, images,texts):
        # Ensure the number of images matches the views_count
        assert len(images) == self.views_count, "Number of images does not match views_count"
        
        view_width, view_height = self.single_view_resolution
        total_width = self.cols * view_width
        total_height = self.rows * view_height
        
        total_image = np.full((total_height, total_width, 3), 255, dtype=np.uint8)
        
        for idx in range(len(images)):

            row = idx // self.cols
            col = idx % self.cols
            single_view_image = self.resize_and_center_image(images[idx],texts[idx])
            total_image[row * view_height:(row + 1) * view_height, col * view_width:(col + 1) * view_width] = single_view_image
        
        return total_image



def make_video(path,config,device="cuda:0",update_handle=print_progress,mano_object=None):
   
    update_handle({'message':"Load data ...",'newline':True})
    sequence = os.path.basename(os.path.abspath(path))
    participant = os.path.basename(os.path.dirname(os.path.abspath(path)))
    print(participant,sequence)
    if participant not in known_participants or sequence not in known_sequences:
        update_handle({'message':"Invalid dataset path",'newline':True})
        return
        # raise ValueError("Invalid dataset path")
    video_save_folder=os.path.join(config.save_video_path,participant)
    if not os.path.exists(video_save_folder):
        os.makedirs(video_save_folder)
    if config.save_images:
        image_save_folder=os.path.join(config.save_images_path,participant,sequence)
        if not os.path.exists(image_save_folder):
            os.makedirs(image_save_folder)

    side=which_side(sequence)
    if side is None:
        update_handle({'message':"Invalid sequence folder name",'newline':True})
        return
        # raise ValueError("Invalid sequence folder name")
    if mano_object is None:
        if config.visibility["mesh"] or config.visibility["joints2D"]:
            mano_object = ManoObject(side=side,device=device)
    if config.visibility["uv_pressure"]:
        mano_obj_path=obj_path[side]
        texcoords, faces_for_uv=parse_obj_for_uv_mapping(mano_obj_path)
        uv_grid=draw_uv_map_with_opencv(texcoords, faces_for_uv, np.min(config.single_view_resolution))

    camera_available = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False}
    for cam_idx in range(8):
        camera_name=cameras[cam_idx]
        if os.path.exists(os.path.join(path,camera_name)):  
            camera_available[cam_idx]=True
    pkl_path=os.path.join(path, "Annotation","annotations")
    indices = sorted([int(f.split('_')[1].split('.')[0]) for f in os.listdir(pkl_path) if f.endswith('.pkl')])
    if config.frame_range[1]==-1 or config.frame_range[1]>len(indices):
        config.frame_range[1]=len(indices)
    if config.frame_range[0]<0:
        config.frame_range[0]=0
    if config.frame_range[0]>=config.frame_range[1]:
        update_handle({'message':"Invalid frame range",'newline':True})
        # raise ValueError("Invalid frame range")

    frame_count=config.frame_range[1]-config.frame_range[0]

    available_cameras_to_show=[cam_idx for cam_idx in range(8) if config.visible_camera_views[cam_idx] and camera_available[cam_idx]]
    views_count=len(available_cameras_to_show)    
    if config.visibility["uv_pressure"]:
        views_count+=1  
    if views_count==0:
        update_handle({'message':"No views selected for visualization",'newline':True})
        #raise ValueError("No views selected for visualization")

    subplot=Subplot(config.single_view_resolution,views_count)
    update_handle({'message':f"Image: {subplot.resolution[0]}x{subplot.resolution[1]}, rows: {subplot.rows}, cols: {subplot.cols}",'newline':True})
    dynamic_camera_pose_file=os.path.join( path, "dynamic_camera_pose.json")
    with open(dynamic_camera_pose_file,'r') as f:
        dynamic_camera_poses = json.load(f)
        dynamic_camera_poses = decompose_and_interpolate(dynamic_camera_poses)
    static_cameras=read_camera_json(os.path.join(path, "config.json"))
    camera_intrinsic_path=os.path.join(path, "config.json")
    with open(camera_intrinsic_path,'r') as f:
        camera_intrinsics = json.load(f)["camera_calibrations"]
    fps=config.fps      
    video_size=(subplot.resolution[0],subplot.resolution[1])
    if config.video_postfix_time:
        vis_name=f"{sequence}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    else:
        vis_name=f"{sequence}"
    out = cv2.VideoWriter(os.path.join(video_save_folder,'{}.mp4'.format(vis_name)),cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)  
    update_handle({'message':"Video rendering started...",'newline':True})
    
    depth_cam_min_max={}
    for index, fid in enumerate(indices[config.frame_range[0]:config.frame_range[1]]):
        
      
        
        force_path=os.path.join(path,"force","{:06}.bin".format(fid))
        pressure=read_pressure_bin(force_path)
        pkl_path=os.path.join(path, "Annotation","annotations","anno_{:06}.pkl".format(fid))
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if config.visibility["mesh"] or config.visibility["joints2D"]:
            batch_size=1
            thetas=torch.from_numpy(data["full_pose"]).to(device).reshape(batch_size,-1)
            betas=torch.from_numpy(data["betas"]).to(device).reshape(batch_size,-1)
            transl=torch.from_numpy(data["transl"]).to(device).reshape(batch_size,-1)
            displacement={"displacement":torch.from_numpy(data["displacement"]).to(device).unsqueeze(0).reshape(batch_size,-1),"normals":torch.from_numpy(data["normals"]).to(device).unsqueeze(0).reshape(batch_size,-1)}
            displacement["normals"]=displacement["normals"].reshape(1,778,3)
            displacement["displacement"]=displacement["displacement"].reshape(1,778,1)*1000
            
            v,_= mano_object(thetas,transl,betas)
            joints=mano_object.get_3d_joints(v)
            disp= displacement["displacement"]*displacement["normals"]/1000
            v_disp=v+disp
            faces=mano_object.mano_mesh.th_faces
            joints_disp=mano_object.get_3d_joints(v_disp)
            

        sub_frames=[]     
        image_texts=[]

        for cam_idx in available_cameras_to_show:
            camera_intrinsic=camera_intrinsics[f'{cam_idx}']
            K = np.array([[camera_intrinsic['fx'], 0, camera_intrinsic['cx']],
                [0, camera_intrinsic['fy'], camera_intrinsic['cy']],
                [0, 0, 1]])
        

            if cam_idx==0:
                image_texts.append({'text':"Ego. View"})
                image_path=os.path.join( path, "camera_d","color","{:06}.jpeg".format(fid)) 
                depth_path=os.path.join( path, "camera_d","depth","{:06}.png".format(fid))
                
                if os.path.exists(image_path) or config.visibility["RGB"]:
                    image=cv2.imread(image_path,cv2.COLOR_BGR2RGB)
                    image_size=image.shape
                    assert image_size[:2]==(camera_intrinsic["ImageSizeY"],camera_intrinsic["ImageSizeX"])



                   
                elif os.path.exists(depth_path) and config.visibility["depth"]:
                    depth=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
                    if cam_idx not in depth_cam_min_max:
                        depth_cam_min_max[cam_idx]=(np.min(depth),np.max(depth))
                    depth_min,depth_max=depth_cam_min_max[cam_idx]
                    image,_,_=apply_colormap_on_depth_image(depth,color_map=config.visualization_config["depth_colormap"],min_val=depth_min,max_val=depth_max)
                    image_size=image.shape
                frame_pose=dynamic_camera_poses["{:06}".format(fid)] 

                    
            else:
                image_texts.append({'text':f"Camera {cam_idx}"})
                image_path=os.path.join( path, f"camera_{cam_idx}","color","{:06}.jpeg".format(fid))
                depth_path=os.path.join( path, f"camera_{cam_idx}","depth","{:06}.png".format(fid))
                if os.path.exists(image_path) and config.visibility["RGB"]:
                    image=cv2.imread(image_path,cv2.COLOR_BGR2RGB)
                    image_size=image.shape
                    assert image_size[:2]==(camera_intrinsic["ImageSizeY"],camera_intrinsic["ImageSizeX"])


                 
                elif os.path.exists(depth_path) and config.visibility["depth"]:
                    depth=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
                    if cam_idx not in depth_cam_min_max:
                        depth_cam_min_max[cam_idx]=(np.min(depth),np.max(depth))
                    depth_min,depth_max=depth_cam_min_max[cam_idx]
                    image,_,_=apply_colormap_on_depth_image(depth,color_map=config.visualization_config["depth_colormap"],min_val=depth_min,max_val=depth_max)
                    image_size=image.shape

                frame_pose=static_cameras[cam_idx]


            R=(frame_pose["R"])
            t=(frame_pose["T"])
            if config.visibility["mesh"] and config.visibility["joints2D"]:
                
                R_torch=torch.from_numpy(R).to(device).float().unsqueeze(0)
                T_torch=torch.from_numpy(t).to(device).float().unsqueeze(0)
                K_torch=torch.from_numpy(K).to(device).float()
            
  
            if config.visibility["wrapped_force"] or config.visibility["sensel_area"]:
                sensel_corner_2D,H =project_rectangle(sensel_corners_3D, K,R,t, scale_factor=1.0)
                sensel_corner_2D=sensel_corner_2D[:,0,:]
                if config.visibility["wrapped_force"]:
                    wrapped_force=get_force_overlay_img(pressure,None,H,image_size, colormap=colormaps[config.visualization_config["force_color_map"]],only_force=True)


   


            if config.visibility["mesh"]:
            
                if config.visibility["vertices_disp"]:
                    vertices_in_camera_frame=get_3d_joints_in_camera_frame(v, R_torch,T_torch)
                else:
                    vertices_in_camera_frame=get_3d_joints_in_camera_frame(v_disp, R_torch,T_torch)  

                focal=((K[0, 0], K[1, 1]),)
                principal_point=((K[0, 2], K[1, 2]),)
                transl_batch=None
                rendered=render_obj_foreground(image_size[:2],vertices_in_camera_frame,transl_batch,focal,principal_point,faces,color=config.visualization_config["mesh_color"])#,color=(0.4882353,  0.3117647,0.25098039 ))

            if config.visibility["joints2D"]:
                
                if config.visibility["vertices_disp"]:
                    j3d_cam=get_3d_joints_in_camera_frame(joints_disp,R_torch,T_torch)
                else:
                    j3d_cam=get_3d_joints_in_camera_frame(joints,R_torch,T_torch)
                j2d=get_2d_points(j3d_cam,K_torch,image_size[:2]).cpu().numpy()[0]
            
                



            if config.visibility["sensel_area"]:
                for i in range(4):
                    cv2.line(image, (int(sensel_corner_2D[i][0]), int(sensel_corner_2D[i][1])), (int(sensel_corner_2D[(i+1)%4][0]), int(sensel_corner_2D[(i+1)%4][1])), (0, 255, 0), 2)
            
            if config.visibility["mesh"]:

                foreground=rendered['foreground'][0]
                mask=rendered['mask']/255
                mask=mask[0,:,:,None]
                alpha=config.visualization_config["mesh_alpha"]
                image=alpha*foreground*mask+ (1-alpha)*image*mask+(1-mask)*image
                image=image.astype(np.uint8)
            
            if config.visibility["wrapped_force"]: 
                image=cv2.addWeighted(image,1.0,wrapped_force,1.0,0)


            if config.visibility["joints2D"]:
                image=draw_skeleton(image,j2d,thickness=config.visualization_config["joint_thickness"])            
            sub_frames.append(image)
        if config.visibility["uv_pressure"]:
            # check whether pressure map is is np.uint8
            uv_pressure_map=data["pressure_map"]
            if uv_pressure_map.dtype==np.uint8:
                uv_pressure_map=uv_pressure_map.astype(float)/255
                         
        
            uv_pressure_map_range=data["pressure_map_range"]
            uv_pressure_map=np.tile(uv_pressure_map,(1,1,3))
            uv_pressure_map=cv2.resize(uv_pressure_map, (uv_grid.shape[0], uv_grid.shape[1]))
            mask = uv_pressure_map > 0
            uv_pressure_map=pressure_to_colormap(uv_pressure_map, colormaps[config.visualization_config["force_color_map"]])
           
            uv_pressure_map = uv_pressure_map * mask
            uv_pressure_map= cv2.addWeighted(uv_grid.copy().astype(int), 1.0, uv_pressure_map.astype(int), 1.0, 0.0)
            sub_frames.append(uv_pressure_map)
            text="Pressure Map"
            image_texts.append({'text':text,'color':(0,0,255)})
        total_view=subplot.create_total_view(sub_frames,image_texts)
        if config.save_images:
            cv2.imwrite(os.path.join(image_save_folder,f"{fid}.png"),total_view)
        

        out.write(total_view)
        update_handle({'message':f"procced frame {index+1}/{frame_count}",'newline':False})

    out.release()
    update_handle({'message':"Video saved",'newline':True})

       
       





        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--sequence_path','-p',type=str,help='Path to the Sequence folder')
    argparser.add_argument('--device','-d',type=str,default='cuda:0',help='Device to run the model on, gpu will accelerate the rendering process')
    # parse video_config
    config=video_config()
    for attr in dir(config):
        if not attr.startswith("__") and not callable(getattr(config,attr)):
            # is dict
            if isinstance(getattr(config,attr),dict):
                if attr=="visible_camera_views":
                    argparser.add_argument(f'--{attr}',type=int,nargs='+',default= [int(key) for key in getattr(config,attr).keys() if getattr(config,attr)[key]],help=config.__hints__[attr])
                else:
                    for key in getattr(config,attr).keys():
                        # mesh color is list 3 input
                        if key=="mesh_color":
                            argparser.add_argument(f'--{attr}_{key}',type=type(getattr(config,attr)[key][0]),nargs=3,default=getattr(config,attr)[key],help=config.__hints__[attr][key])
                        else:
                            argparser.add_argument(f'--{attr}_{key}',type=type(getattr(config,attr)[key]),default=getattr(config,attr)[key],help=config.__hints__[attr][key])
            # is list
            elif isinstance(getattr(config,attr),list):
                argparser.add_argument(f'--{attr}',type=type(getattr(config,attr)[0]),nargs='+',default=getattr(config,attr),help=config.__hints__[attr])
            else:
                argparser.add_argument(f'--{attr}',type=type(getattr(config,attr)),default=getattr(config,attr),help=config.__hints__[attr])
    args = argparser.parse_args()
    for attr in dir(config):
        if not attr.startswith("__") and not callable(getattr(config,attr)):    
            if isinstance(getattr(config,attr),dict):
                if attr=="visible_camera_views":
                    for key in getattr(config,attr):
                        if key in getattr(args,attr):
                            getattr(config,attr)[key]=True
                        else:
                            getattr(config,attr)[key]=False
                for key in getattr(config,attr).keys():
                    if hasattr(args,f"{attr}_{key}"):
                        getattr(config,attr)[key]=getattr(args,f"{attr}_{key}")
            elif hasattr(args,attr):
               config.Set(attr,getattr(args,attr))
    print("config:")
    config.display()
    make_video(args.sequence_path,config,device=args.device,update_handle=print_progress,mano_object=None)
