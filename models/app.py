import os
import cv2
import json
import argparse
import numpy as np
import pickle
import torch
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog,messagebox,simpledialog
from tkinter import StringVar, IntVar, BooleanVar, DoubleVar

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from models.sensel_projection import project_rectangle, sensel_corners_3D
from models.visualization import render_obj_foreground, get_2d_points, get_3d_joints_in_camera_frame,draw_skeleton,ManoObject,apply_colormap_on_depth_image
from models.pressure_util import  get_force_overlay_img
from models.io import read_pressure_bin,which_side,decompose_and_interpolate,read_camera_json
from models.io import participants as known_participants
from models.io import sequences as known_sequences
from models.io import cameras,colormaps
from models.make_video import make_video,video_config
class Visualizer:
    def __init__(self, master,base_path,device):
        self.device=device
        self.hint_below="rightarrow: next frame, leftarrow: previous frame \n"
        self.hint_below+="0(EgoView),1,2,3,4,5,6,7(camera view) \n"
        self.hint_below+="q: RGB/Depth, w: show mesh, e: vertices_disp, r: pressure overlay, t: sensel area, z: joints2D\n"
        self.visualization_config={"depth_colormap":"GRAY","force_color_map":"OCEAN","mesh_color":(0.4882353,  0.3117647,0.25098039 ),"mesh_alpha":0.5,"joint_thickness":2}
     
        self.master = master
        self.right_mano_object=ManoObject(side="right",device=device)
        self.left_mano_object=ManoObject(side="left",device=device)

        # Load images

        self.base_path = base_path
        self.loaded_data=None
        
        
  
        
   
        #print("Images loaded:", self.image_files, self.segmentation_files)
       


        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.fig.text(0.5, -0.01, self.hint_below, ha='center', va='bottom', fontsize=10)

        self.ax = self.fig.add_subplot()
        

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)


        # Bind events
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

        self.set_visibility()
        self.camera_available = {0: True, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False}
        self.get_available_camera()
        self.get_current_frame()


    
    def load_camera_view(self,loaded_data,cam_idx,fid):
        if cam_idx not in loaded_data["images"]:
       
            camera_intrinsic=loaded_data["camera_intrinsic"][f'{cam_idx}']
            K = np.array([[camera_intrinsic['fx'], 0, camera_intrinsic['cx']],
                    [0, camera_intrinsic['fy'], camera_intrinsic['cy']],
                    [0, 0, 1]])
            loaded_data["camera_intrinsic"][cam_idx]=K
   
            if cam_idx==0:
                image_path=os.path.join( loaded_data["path"], "camera_d","color","{:06}.jpeg".format(fid)) 
                depth_path=os.path.join( loaded_data["path"], "camera_d","depth","{:06}.png".format(fid))
                if os.path.exists(image_path):
                    loaded_data["images"][cam_idx]=cv2.imread(image_path,cv2.COLOR_BGR2RGB)
                    image_size=loaded_data["images"][cam_idx].shape
                    assert image_size[:2]==(camera_intrinsic["ImageSizeY"],camera_intrinsic["ImageSizeX"])

                    dynamic_camera_pose_file=os.path.join( loaded_data["path"], "dynamic_camera_pose.json")
                    with open(dynamic_camera_pose_file,'r') as f:
                        camera_poses = json.load(f)
                        camera_pose=decompose_and_interpolate(camera_poses)
                        frame_pose=camera_pose["{:06}".format(fid)] 
                        R=(frame_pose["R"])
                        t=(frame_pose["T"])
                        loaded_data["camera_pose"][cam_idx]={'R':R,'t':t}
                if os.path.exists(depth_path):
                    loaded_data["depth"][cam_idx]=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)
                    #apply colormap
                    
            else:
                image_path=os.path.join( loaded_data["path"], f"camera_{cam_idx}","color","{:06}.jpeg".format(fid))
                depth_path=os.path.join( loaded_data["path"], f"camera_{cam_idx}","depth","{:06}.png".format(fid))
                if os.path.exists(image_path):
                    loaded_data["images"][cam_idx]=cv2.imread(image_path,cv2.COLOR_BGR2RGB)
                    image_size=loaded_data["images"][cam_idx].shape
                    assert image_size[:2]==(camera_intrinsic["ImageSizeY"],camera_intrinsic["ImageSizeX"])
                    
                    
                    frame_pose=loaded_data["static_cameras"][cam_idx]
                    R=(frame_pose["R"])
                    t=(frame_pose["T"])
                    loaded_data["camera_pose"][cam_idx]=frame_pose
                if os.path.exists(depth_path):
                    loaded_data["depth"][cam_idx]=cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)



            sensel_corner_2D,H =project_rectangle(sensel_corners_3D, K,R,t, scale_factor=1.0)
            loaded_data["H"][cam_idx]=H
            pressure=loaded_data["force"]
        
            wrapped_force=get_force_overlay_img(pressure,None,H,image_size, colormap=colormaps[self.visualization_config["force_color_map"]],only_force=True)



            loaded_data["wrapped_force"][cam_idx]=wrapped_force
            loaded_data["sensel_corners_2D"][cam_idx]=sensel_corner_2D

            R_torch=torch.from_numpy(R).to(self.device).float().unsqueeze(0)
            T_torch=torch.from_numpy(t).to(self.device).float().unsqueeze(0)
            K_torch=torch.from_numpy(K).to(self.device).float()
            
            v=loaded_data["vertices"]
            v_disp=loaded_data["vertices_disp"]
            joints=loaded_data["joints"]
            joints_disp=loaded_data["joints_disp"]

            vertices_disp_in_camera_frame=get_3d_joints_in_camera_frame(v_disp, R_torch,T_torch)
            
            vertices_in_camera_frame=get_3d_joints_in_camera_frame(v, R_torch,T_torch)
            
     
             
            j3d_cam=get_3d_joints_in_camera_frame(joints,R_torch,T_torch)
            j2d=get_2d_points(j3d_cam,K_torch,image_size[:2]).cpu().numpy()[0]
            loaded_data["joints2D"][cam_idx]=j2d

            j3d_disp_cam=get_3d_joints_in_camera_frame(joints_disp,R_torch,T_torch)
            j2d_disp=get_2d_points(j3d_disp_cam,K_torch,image_size[:2]).cpu().numpy()[0]
            loaded_data["joints2D_disp"][cam_idx]=j2d_disp


            focal=((K[0, 0], K[1, 1]),)
            principal_point=((K[0, 2], K[1, 2]),)
            transl_batch=None
            faces=loaded_data["render_settings"]["faces"]
        
     
            rendered=render_obj_foreground(image_size[:2],vertices_in_camera_frame,transl_batch,focal,principal_point,faces,color=self.visualization_config["mesh_color"])#,color=(0.4882353,  0.3117647,0.25098039 ))
            rendered_disp=render_obj_foreground(image_size[:2],vertices_disp_in_camera_frame,transl_batch,focal,principal_point,faces,color=self.visualization_config["mesh_color"])
            loaded_data["rendered"][cam_idx]=rendered
            loaded_data["rendered_disp"][cam_idx]=rendered_disp
            loaded_data["render_settings"][cam_idx]={"image_size":image_size,"focal":focal,"principal_point":principal_point,"vertices":vertices_in_camera_frame,"vertices_disp":vertices_disp_in_camera_frame}
            
        return loaded_data
    
    def load_data(self,current_image=None):


        if current_image is None:
            current_image=self.current_image
        if self.current_image >= len(self.indices) or self.current_image<0:
            return
        
        
        fid=self.indices[current_image]
        loaded_data={}
        path=self.folder_path
        loaded_data["path"]=path
        force_path=os.path.join(path,"force","{:06}.bin".format(fid))
        pkl_path=os.path.join(path, "Annotation","annotations","anno_{:06}.pkl".format(fid))
        
        loaded_data["force"]=read_pressure_bin(force_path)
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        loaded_data["annotation"]=data
        loaded_data["depth"]={}
        loaded_data["images"]={}
        loaded_data["camera_pose"]={}
        loaded_data["wrapped_force"]={}
        loaded_data["sensel_corners_2D"]={}
        loaded_data["rendered"]={}
        loaded_data["rendered_disp"]={}
        loaded_data["joints2D"]={}
        loaded_data["joints2D_disp"]={}
        batch_size=1

        thetas=torch.from_numpy(data["full_pose"]).to(self.device).reshape(batch_size,-1)

        betas=torch.from_numpy(data["betas"]).to(self.device).reshape(batch_size,-1)
        transl=torch.from_numpy(data["transl"]).to(self.device).reshape(batch_size,-1)
        displacement={"displacement":torch.from_numpy(data["displacement"]).to(self.device).unsqueeze(0).reshape(batch_size,-1),"normals":torch.from_numpy(data["normals"]).to(self.device).unsqueeze(0).reshape(batch_size,-1)}
        displacement["normals"]=displacement["normals"].reshape(1,778,3)
        displacement["displacement"]=displacement["displacement"].reshape(1,778,1)*1000
        
        if self.hand_side=="right":
            mano_object=self.right_mano_object
        elif self.hand_side=="left":
            mano_object=self.left_mano_object
        else:
            raise ValueError("Invalid hand side")
        

        v,_= mano_object(thetas,transl,betas)
        joints=mano_object.get_3d_joints(v)
        disp= displacement["displacement"]*displacement["normals"]/1000
        v_disp=v+disp
        joints_disp=mano_object.get_3d_joints(v_disp)
        loaded_data["joints"]=joints
        loaded_data["joints_disp"]=joints_disp
        loaded_data["vertices"]=v
        loaded_data["vertices_disp"]=v_disp
        faces=mano_object.mano_mesh.th_faces
        loaded_data["render_settings"]={}
        loaded_data["render_settings"]["faces"]=faces
        loaded_data["camera_intrinsic"]={}
        loaded_data["H"]={}

        static_cameras=read_camera_json(os.path.join(path, "config.json"))  
        loaded_data["static_cameras"]=static_cameras
        camera_intrinsic_path=os.path.join(path, "config.json")
        with open(camera_intrinsic_path,'r') as f:
            camera_intrinsics = json.load(f)["camera_calibrations"]
        loaded_data["camera_intrinsic"]=camera_intrinsics

        cam_idx=self.selected_camera.get()
        
        loaded_data=self.load_camera_view(loaded_data,cam_idx,fid)
        self.loaded_data=loaded_data
    
    def set_visualization_config(self,kargs):
        rerender_mesh=False
        rerender_force=False
        for key, value in kargs.items():
            if key in self.visualization_config:
                if key=="mesh_color" and value!=self.visualization_config[key]:
                    rerender_mesh=True
                if key=="force_color_map" and value!=self.visualization_config[key]:
                    rerender_force=True

                self.visualization_config[key] = value
            else:
                raise ValueError(f"Invalid key: {key}")
            
            self.display_image(rerender_mesh=rerender_mesh,rerender_force=rerender_force)
    
    def redirec_to_folder(self,selected_participant,selected_sequence):
        self.folder_path=os.path.join(self.base_path,selected_participant,selected_sequence)
        
        for cam_idx in range(8):
            camera_name=cameras[cam_idx]
            if os.path.exists(os.path.join(self.folder_path,camera_name)):
                self.camera_available[cam_idx]=True


        self.hand_side=which_side(selected_sequence)
        pkl_path=os.path.join(self.folder_path, "Annotation","annotations")
        
        self.indices = sorted([int(f.split('_')[1].split('.')[0]) for f in os.listdir(pkl_path) if f.endswith('.pkl')])
        self.hint =f" current image:{'{}'}--{'{}'}/{len(self.indices)}\n"
        self.current_image=0  
        self.load_data()
        self.update_camera_combobox()
        self.update_fid_combobox()
      
        self.display_image()
    


    def display_image(self,rerender_mesh=False,rerender_force=False):

        cam_id=self.selected_camera.get()
        if cam_id in self.loaded_data["images"]:
            if rerender_force:
                key=cam_id
                if key in self.loaded_data["wrapped_force"]:
                    image_size=self.loaded_data["render_settings"][key]["image_size"]
                    self.loaded_data["wrapped_force"][key]=get_force_overlay_img(self.loaded_data["force"],self.loaded_data["images"][key],self.loaded_data["H"][key],image_size, colormap=colormaps[self.visualization_config["force_color_map"]],only_force=True)

            if rerender_mesh:
                key=cam_id
                if key in self.loaded_data["rendered"]:
                    image_size=self.loaded_data["render_settings"][key]["image_size"]
                    vertices=self.loaded_data["render_settings"][key]["vertices"]
                    vertices_disp=self.loaded_data["render_settings"][key]["vertices_disp"]
                    faces=self.loaded_data["render_settings"]["faces"]
                    focal=self.loaded_data["render_settings"][key]["focal"]
                    principal_point=self.loaded_data["render_settings"][key]["principal_point"]
                    transl_batch=None
                    self.loaded_data["rendered"][key]=render_obj_foreground(image_size,vertices,transl_batch,focal,principal_point,faces,color=self.visualization_config["mesh_color"])  
                    self.loaded_data["rendered_disp"][key]=render_obj_foreground(image_size,vertices_disp,transl_batch,focal,principal_point,faces,color=self.visualization_config["mesh_color"])
        else:
            self.loaded_data=self.load_camera_view(self.loaded_data,cam_id,self.indices[self.current_image])

        if self.loaded_data is None:
            return
        if self.visibility["RGB"]:
            self.img = self.loaded_data["images"][cam_id].copy()
        elif self.visibility["depth"]:
            self.img = self.loaded_data["depth"][cam_id].copy()
            self.img,self.depth_min_val,self.depth_max_val = apply_colormap_on_depth_image(self.img,self.visualization_config["depth_colormap"])
        else:
            raise ValueError("Invalid visibility state")
        
        if self.visibility["wrapped_force"]:
            wrapped_force=self.loaded_data["wrapped_force"][cam_id]
            self.img=cv2.addWeighted(self.img,1.0,wrapped_force,1.0,0)

        if self.visibility["sensel_area"]:
            sensel_corner_2D=self.loaded_data["sensel_corners_2D"][cam_id][:,0,:]
            for i in range(4):
                cv2.line(self.img, (int(sensel_corner_2D[i][0]), int(sensel_corner_2D[i][1])), (int(sensel_corner_2D[(i+1)%4][0]), int(sensel_corner_2D[(i+1)%4][1])), (0, 255, 0), 2)
        if self.visibility["mesh"]:
            if self.visibility["vertices_disp"]:
                rendered=self.loaded_data["rendered_disp"][cam_id]
            else:
                rendered=self.loaded_data["rendered"][cam_id]
            foreground=rendered['foreground'][0]
            mask=rendered['mask']/255
            mask=mask[0,:,:,None]
            alpha=self.visualization_config["mesh_alpha"]
            self.img=alpha*foreground*mask+ (1-alpha)*self.img*mask+(1-mask)*self.img 
            self.img=self.img.astype(np.uint8)

        if self.visibility["joints2D"]:
            joints2D=self.loaded_data["joints2D"][cam_id]
            self.img=draw_skeleton(self.img,joints2D,thickness=self.visualization_config["joint_thickness"])            

        self.redraw_image()

    def redraw_image(self,hint=""):
        self.ax.clear()
        self.ax.remove()  
        self.ax = self.fig.add_subplot()         
        if hasattr(self, 'cbar') and self.cbar:
            self.cbar.remove()
            self.cbar = None
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img)
        if self.visibility["depth"]:
            norm = Normalize(vmin=self.depth_min_val, vmax=self.depth_max_val)
            cmap_name = self.visualization_config["depth_colormap"]
            sm = cm.ScalarMappable(cmap=cm.get_cmap(cmap_name.lower()), norm=norm)
            sm.set_array([])

            self.cbar = self.fig.colorbar(sm, ax=self.ax, orientation='vertical', fraction=0.046, pad=0.04)
            self.cbar.set_label('Depth(mm)')      

        self.ax.set_title(self.hint.format('{:06}'.format(self.indices[self.current_image]),self.current_image+1)+hint)
        
        self.canvas.draw()
        
    def move_to_next_image(self):
        if self.current_image<len(self.indices)-1:      
            self.current_image += 1
            self.load_data()
            self.display_image()
        else:
          
            self.redraw_image(hint)
   
    def set_visibility(self):
        self.visibility = {
            "RGB": True,
            "depth": False,
            "mesh": False,
            "vertices_disp": False,
            "wrapped_force": False,
            "sensel_area": False,
            "joints2D": False
        }

        # Frame for the buttons
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Radio buttons for mutually exclusive RGB and depth
        self.rgb_depth_var = tk.StringVar(value="RGB")

        self.radio_rgb = tk.Radiobutton(self.button_frame, text="RGB", variable=self.rgb_depth_var, value="RGB", command=self.update_visibility)
        self.radio_rgb.pack(side=tk.LEFT, padx=5, pady=5)

        self.radio_depth = tk.Radiobutton(self.button_frame, text="Depth", variable=self.rgb_depth_var, value="depth", command=self.update_visibility)
        self.radio_depth.pack(side=tk.LEFT, padx=5, pady=5)

        # Checkboxes for other options
        self.checkbuttons = {}
        checkbox_keys = ["mesh", "vertices_disp", "wrapped_force", "sensel_area", "joints2D"]
        keys = ['w', 'e', 'r', 't', 'z']

        for key, checkbox_key in zip(keys, checkbox_keys):
            var = tk.BooleanVar(value=self.visibility[checkbox_key])
            self.checkbuttons[checkbox_key] = var
            cb = tk.Checkbutton(self.button_frame, text=checkbox_key, variable=var, command=self.update_visibility)
            cb.pack(side=tk.LEFT, padx=5, pady=5)
            # Bind keys to checkboxes
            self.master.bind(key, lambda event, k=checkbox_key: self.toggle_checkbox(k))

        # Bind the 'q' key to toggle between RGB and depth
        self.master.bind('q', self.toggle_rgb_depth)        
        
    def get_available_camera(self):
        self.selected_camera = tk.IntVar(value=0)

        self.available_cameras = [camera for camera, available in self.camera_available.items() if available]
        self.available_camera_names = [cameras[camera] for camera in self.available_cameras]
        self.camera_label = tk.Label(self.button_frame, text="View:")
        self.camera_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.camera_combobox = ttk.Combobox(self.button_frame, values=self.available_camera_names, state='readonly')
        self.camera_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        self.camera_combobox.bind("<<ComboboxSelected>>", self.update_selected_camera)
        self.camera_combobox.current(0)  # Set the default selected value to the first available camera
   
    def get_current_frame(self):
        self.fid_label = tk.Label(self.button_frame, text="Frame:")
        self.fid_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.fid_combobox = ttk.Combobox(self.button_frame, state='readonly')
        self.fid_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        self.fid_combobox.bind("<<ComboboxSelected>>", self.update_selected_fid)

    def update_selected_fid(self, event=None):
        # Update the selected camera based on the combobox selection
        selected_fid = int(self.fid_combobox.get())
    
        current_image=self.indices.index(selected_fid)
        if current_image != self.current_image:
            self.current_image = current_image
            self.load_data()
            self.display_image()
   
    def update_fid_combobox(self):
        # Update the combobox items based on camera availability
        self.available_fids = ["{:06}".format(fid) for fid in self.indices]
        self.fid_combobox['values'] = self.available_fids
      
        # Set default selected value to the first available camera
        if self.available_fids:
            self.fid_combobox.current(0)
            self.update_selected_fid()
 
    def move_to_previous_image(self):
        if self.current_image > 0:
            self.current_image -= 1
            self.load_data()
            self.display_image()
        else:
            hint="first image"
   
            self.redraw_image(hint)
    
    def update_visibility(self):
        # Update visibility dictionary for radio buttons
        self.visibility["RGB"] = (self.rgb_depth_var.get() == "RGB")
        self.visibility["depth"] = (self.rgb_depth_var.get() == "depth")
        
                
        # mesh_visibility= self.checkbuttons["mesh"].get() 
        # if mesh_visibility == False and self.visibility["mesh"]==True:
        #     self.checkbuttons["vertices_disp"].set(False)
        # if mesh_visibility == self.visibility["mesh"]:
        #     self.checkbuttons["mesh"].set(self.visibility["mesh"] or self.checkbuttons["vertices_disp"].get())

        # Update visibility dictionary for checkboxes
        for key, var in self.checkbuttons.items():
            self.visibility[key] = var.get()

        # Print the updated visibility states
        
        self.display_image()

    def toggle_rgb_depth(self, event=None):
        # Toggle between RGB and depth
        if self.rgb_depth_var.get() == "RGB":
            self.rgb_depth_var.set("depth")
        else:
            self.rgb_depth_var.set("RGB")
        self.update_visibility()

    def toggle_checkbox(self, key):
        # Toggle the checkbox
        current_value = self.checkbuttons[key].get()
        self.checkbuttons[key].set(not current_value)
        self.update_visibility()

    def update_camera_combobox(self):
        # Update the combobox items based on camera availability
        self.available_cameras = [camera for camera, available in self.camera_available.items() if available]
        self.available_camera_names = [cameras[camera] for camera in self.available_cameras]
        self.camera_combobox['values'] = self.available_camera_names

        # Set default selected value to the first available camera
        if self.available_camera_names:
            self.camera_combobox.current(0)
            self.update_selected_camera()

    def update_selected_camera(self, event=None):
        # Update the selected camera based on the combobox selection
        selected_camera_name = self.camera_combobox.get()
        for key, name in cameras.items():
            if name == selected_camera_name:
                self.selected_camera.set(key)
                break
        self.display_image()
    
    def on_key_press(self, event):
        
        cameras_keys = ['0', '1', '2', '3', '4', '5', '6', '7']
        if event.key in cameras_keys:
            if int(event.key) in cameras and int(event.key) in self.available_cameras:	
                self.camera_combobox.set(cameras[int(event.key)])
                self.selected_camera.set(int(event.key))
                self.display_image()
        elif event.key == 'left':
            self.move_to_previous_image()
        elif event.key == 'right':
            self.move_to_next_image()
class Application:
    def __init__(self, master,base_path,device="cuda:0"):
        self.master = master
        self.master.title("EgoPressure Visualizer")
        self.create_menu()
        self.create_status_bar()
        self.base_path=base_path
        self.device=device
        self.visualizer = Visualizer(self.master,base_path,device) 

        self.selected_participant = None
        self.selected_sequence = None
        self.video_generation_config=video_config()
        self.video_dialog =VideoConfigDialog(self.video_generation_config)
        self.video_dialog_window = None
        self.processing_video=False
    
  
    def get_participants_from_base_path(self):
        folder_list= os.listdir(self.base_path)
        filtered_list = [folder for folder in folder_list if folder in known_participants]
        return filtered_list

    def get_sequences_from_participant(self,participant):
        folder_list= os.listdir(os.path.join(self.base_path,participant))
        filtered_list = [folder for folder in folder_list if folder in known_sequences]

        return filtered_list

    def create_menu(self):
        # Create a menu bar
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        # Create a 'File' menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Recording", command=self.load_recording_dialog)
        file_menu.add_command(label="Export Video", command=self.export_video_dialog)
        config_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Config", menu=config_menu)
        config_menu.add_command(label="Visualization Config", command=self.open_visualization_config)

    def create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
   
    def open_recording(self):
        if self.selected_participant and self.selected_sequence:
            self.visualizer.redirec_to_folder(self.selected_participant,self.selected_sequence)

    def load_recording_dialog(self):
        dialog = tk.Toplevel(self.master)
        dialog.title("Open Recording")
        self.selected_participant = None
        self.selected_sequence = None
     
        def store_selections():
            self.selected_participant= self.combo_participant.get()
            self.selected_sequence = self.combo_sequence.get()
            if not self.selected_participant or not self.selected_sequence:
                self.update_status("Please select a participant and a sequence")
                self.selected_participant = None
                self.selected_sequence = None
                
            else:
                self.update_status(f'Open - Participant: {self.selected_participant}, Sequence: {self.selected_sequence}')
            
            dialog.destroy()
            self.open_recording()
        

        tk.Label(dialog, text="Participant:").pack(pady=5)
        self.combo_participant = ttk.Combobox(dialog)
        self.combo_participant.config(width=40)
        self.combo_participant['values'] = self.get_participants_from_base_path()
        if self.combo_participant['values']:
            self.combo_participant.current(0)
        self.combo_participant.pack(pady=5)
        self.combo_participant.bind("<<ComboboxSelected>>", self.on_select_combo_participant)
        


        tk.Label(dialog, text="Sequence:").pack(pady=5)
        self.combo_sequence = ttk.Combobox(dialog)
        self.combo_sequence.config(width=40)    
        self.combo_sequence['values'] = self.get_sequences_from_participant(self.combo_participant.get())
        self.combo_sequence.current(0)
        self.combo_sequence.pack(pady=5)
        self.combo_sequence.bind("<<ComboboxSelected>>", self.on_select_combo_sequence)
        

        okay_button = tk.Button(dialog, text="Okay", command=store_selections)
        okay_button.pack(pady=10)
        self.update_status("Opened load recording dialog")

    def on_select_combo_participant(self, event):
        selected_participant = self.combo_participant.get()
        self.combo_sequence['values'] = self.get_sequences_from_participant(selected_participant)
        if self.combo_sequence['values']:
            self.combo_sequence.current(0)
        self.update_status(f'Selected Participant : {selected_participant}')

    def on_select_combo_sequence(self, event):
        selected_sequence = self.combo_sequence.get()
        self.update_status(f'Selected Sequence: {selected_sequence}')

    def update_status(self, message):
        if isinstance(message, dict):
            message = message['message']
        self.status_var.set(message)
    
    def open_visualization_config(self):

        dialog = tk.Toplevel(self.master)
        dialog.title("Visualization Config")

        tk.Label(dialog, text="Depth Colormap:").pack(pady=5)
        self.depth_colormap = ttk.Combobox(dialog)
        self.depth_colormap['values'] = list(colormaps.keys())
        self.depth_colormap.set(self.visualizer.visualization_config.get("depth_colormap", "GRAY"))
        self.depth_colormap.pack(pady=5)
        self.depth_colormap.bind("<<ComboboxSelected>>", lambda e: self.update_config('depth_colormap', self.depth_colormap.get()))

        tk.Label(dialog, text="Force Colormap:").pack(pady=5)
        self.force_colormap = ttk.Combobox(dialog)
        
   
        force_color_map=list(colormaps.keys())
        force_color_map.remove('JET')
        self.force_colormap['values'] = force_color_map
        self.force_colormap.set(self.visualizer.visualization_config.get("force_color_map", "OCEAN"))
        self.force_colormap.pack(pady=5)
        self.force_colormap.bind("<<ComboboxSelected>>", lambda e: self.update_config('force_color_map', self.force_colormap.get()))

        tk.Label(dialog, text="Mesh Color:").pack(pady=5)
        self.mesh_color_frame = tk.Frame(dialog)
        self.mesh_color_frame.pack(pady=5)
        self.mesh_color_r = self.create_entry_with_label(self.mesh_color_frame, "R", self.visualizer.visualization_config["mesh_color"][0] * 255)
        self.mesh_color_g = self.create_entry_with_label(self.mesh_color_frame, "G", self.visualizer.visualization_config["mesh_color"][1] * 255)
        self.mesh_color_b = self.create_entry_with_label(self.mesh_color_frame, "B", self.visualizer.visualization_config["mesh_color"][2] * 255)

        tk.Label(dialog, text="Mesh Alpha:").pack(pady=5)
        self.mesh_alpha = self.create_slider_with_entry(dialog, 0, 1, self.visualizer.visualization_config["mesh_alpha"])

        tk.Label(dialog, text="Joint Thickness:").pack(pady=5)
        self.joint_thickness = self.create_entry_with_label(dialog, "joint_thickness", self.visualizer.visualization_config["joint_thickness"], vertical=True,enable_label=False)


        okay_button = tk.Button(dialog, text="Okay", command=lambda: self.on_visconfig_okay_button(dialog))
        okay_button.pack(pady=10)

        self.update_status("Opened visualization config dialog")
    def on_visconfig_okay_button(self,dialog):
        self.visualizer.visualization_config["joint_thickness"] = int(self.joint_thickness.get())
        self.visualizer.visualization_config["mesh_color"] = (int(self.mesh_color_r.get()) / 255, int(self.mesh_color_g.get()) / 255, int(self.mesh_color_b.get()) / 255)
        self.visualizer.visualization_config["mesh_alpha"] = float(self.mesh_alpha.get())
        self.visualizer.visualization_config["force_color_map"] = self.force_colormap.get()
        self.visualizer.visualization_config["depth_colormap"] = self.depth_colormap.get()
        self.visualizer.display_image(rerender_mesh=True,rerender_force=True)
        dialog.destroy()
    def create_entry_with_label(self, parent, label_text, initial_value,vertical=False,enable_label=True):
        frame = tk.Frame(parent)
        if vertical:
            frame.pack(pady=5)
        else:
            frame.pack(side=tk.LEFT, padx=5)
        if enable_label:
            tk.Label(frame, text=label_text).pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=5)
        entry.insert(0, str(int(initial_value)))
        entry.pack(side=tk.LEFT)
        entry.bind('<FocusOut>', lambda e: self.update_config_entry(label_text.lower(), entry))        
        entry.bind('<Return>', lambda e: self.update_config_entry(label_text.lower(), entry))

        return entry

    def create_slider_with_entry(self, parent, from_, to, initial_value):
        frame = tk.Frame(parent)
        frame.pack(pady=5)
        slider = tk.Scale(frame, from_=from_, to=to, resolution=0.01, orient=tk.HORIZONTAL,showvalue=False)
        slider.set(initial_value)
        slider.pack(side=tk.LEFT)
        entry = tk.Entry(frame, width=5)
        entry.insert(0, str(initial_value))
        entry.pack(side=tk.LEFT)
        slider.config(command=lambda val: self.update_config_slider('mesh_alpha', val, entry))
        entry.bind('<Return>', lambda e: slider.set(float(entry.get())))
        entry.bind('<KeyRelease>', lambda e: self.update_config_slider('mesh_alpha', entry.get(), slider))
        return slider

    def update_config(self, key, value):
        self.visualizer.set_visualization_config({key: value})
        self.update_status(f'Updated {key}: {value}')

    def update_config_entry(self, key, entry):
        try:
            value = int(entry.get())
            if key in ['r', 'g', 'b']:
                mesh_color = list(self.visualizer.visualization_config['mesh_color'])
                index = {'r': 0, 'g': 1, 'b': 2}[key]
                mesh_color[index] = value / 255
                self.visualizer.set_visualization_config({'mesh_color': tuple(mesh_color)})
                
                
                self.update_status(f'Updated mesh_color: {self.visualizer.visualization_config["mesh_color"]}')
            else:
                self.visualizer.set_visualization_config({key: value})
                self.update_status(f'Updated {key}: {value}')
        except ValueError:
            pass

    def update_config_slider(self, key, value, entry_or_slider):
        try:
            value = float(value)
            self.visualizer.set_visualization_config({key: value})
            self.update_status(f'Updated {key}: {value}')
            if isinstance(entry_or_slider, tk.Entry):
                entry_or_slider.delete(0, tk.END)
                entry_or_slider.insert(0, str(value))
            else:
                entry_or_slider.set(value)
        except ValueError:
            pass
    
    def export_video_dialog(self):
        


        if self.video_dialog_window is not None:
            self.video_dialog_window.destroy()
            self.video_dialog_window = None
        self.update_status("Open export video dialog")

        self.video_dialog_window = tk.Toplevel(self.master)
        self.video_dialog_window.title("Export Video")
        self.video_dialog_window_open=True
        self.video_dialog_window.protocol("WM_DELETE_WINDOW", self.on_video_dialog_close)
        available_cameras=[cam_i for cam_i in self.visualizer.camera_available.keys() if self.visualizer.camera_available[cam_i]]
        row = self.video_dialog.make_dialog(self.video_dialog_window,available_cameras)

        info_frame=tk.LabelFrame(self.video_dialog_window,text="Info", padx=5, pady=5)
        info_frame.grid(row=row, column=0, padx=10, pady=5, sticky=tk.W)
        participant_label = tk.Label(info_frame, text="Participant: {}".format("demo_participant"))     #self.selected_participant
        participant_label.grid(row=0, column=0, sticky=tk.W)
        sequence_label = tk.Label(info_frame, text="Sequence: {}".format(self.selected_sequence))
        sequence_label.grid(row=1, column=0, sticky=tk.W)
 
        button_frame = tk.Frame(self.video_dialog_window)
        button_frame.grid(row=row+1, column=0, pady=10)
        
        self.export_video_button = tk.Button(button_frame, text="Export Video", command=self.on_export_video_button)
        self.export_video_button.grid(row=row+1, column=0, padx=5)
        if self.processing_video:
            self.export_video_button.config(state=tk.DISABLED)
        if not self.selected_participant or not self.selected_sequence:
            self.export_video_button.config(state=tk.DISABLED)


    def on_video_dialog_close(self):
        self.video_dialog_window_open = False
        self.video_dialog_window.destroy()
        self.video_dialog_window = None
    


    def on_export_video_button(self):
        self.video_dialog.save()
        self.start_video_thread()
        self.update_status("Exporting video...")
        self.video_dialog_window.destroy()  # Close the dialog
        self.video_dialog_window_open = False
        self.video_dialog_window = None

    def video_background_task(self, callback):
        path=os.path.join(self.base_path,self.selected_participant,self.selected_sequence)
        side=which_side(self.selected_sequence)
        if side=="right":
            mano_object=self.visualizer.right_mano_object
        elif side=="left":
            mano_object=self.visualizer.left_mano_object
        else:
            self.update_status("Invalid hand side")
            callback()
            return
        make_video(path, self.video_dialog.config,self.device,self.update_status,mano_object=mano_object)
        callback()
    def start_video_thread(self):
        self.processing_video = True
        self.export_video_button.config(state=tk.DISABLED)  # Disable the button
        def on_thread_complete():
            self.processing_video = False
            if self.video_dialog_window_open:
                self.export_video_button.config(state=tk.NORMAL)  # Re-enable the button
            
        thread = threading.Thread(target=self.video_background_task, args=(on_thread_complete,))
        thread.start()



class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="yellow", relief="solid", borderwidth=1)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
def enforce_int_limit(event, variable):

    try:
        value = int(variable.get())
        if value < 0:
            variable.set(0)
        elif value > 255:
            variable.set(255)
    except:
        variable.set(0)

class VideoConfigDialog:
    def __init__(self, config):
        self.config = config

    def make_dialog(self,dialog,available_cameras=[0,1,2,3,4,5,6,7]):
       
        # Video configuration frame
        video_config_frame = tk.LabelFrame(dialog,text="Video Configuration", padx=5, pady=5)
        video_config_frame.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)

        # FPS entry
        fps_label = tk.Label(video_config_frame, text="FPS")
        fps_label.grid(row=0, column=0, sticky=tk.W)
        self.fps = IntVar(value=self.config.fps)
        fps_entry = tk.Entry(video_config_frame, textvariable=self.fps,width=10,justify=tk.CENTER	)
        fps_entry.grid(row=0, column=1, sticky=tk.W)
        ToolTip(fps_label, self.config.__hints__["fps"])

        # Frame range entries

        frame_range_label = tk.Label(video_config_frame, text="Frame Range")
        frame_range_label.grid(row=1, column=0, sticky=tk.W)
        self.frame_range = [IntVar(value=val) for val in self.config.frame_range]
        frame_range_entries = []
        frame_entry_frame=tk.Frame(video_config_frame)
        frame_entry_frame.grid(row=1, column=1, sticky=tk.W)
        for i in range(2):
            entry = tk.Entry(frame_entry_frame, textvariable=self.frame_range[i],width=5,justify=tk.CENTER)
            entry.grid(row=0, column=i)
            frame_range_entries.append(entry)
        ToolTip(frame_range_label, self.config.__hints__["frame_range"])

        # Single view resolution entries
        single_view_resolution_label = tk.Label(video_config_frame, text="Single View Resolution")
        single_view_resolution_label.grid(row=2, column=0, sticky=tk.W)
        self.single_view_resolution = [IntVar(value=val) for val in self.config.single_view_resolution]
        single_view_resolution_entries = []
        single_view_resolution_frame=tk.Frame(video_config_frame)
        single_view_resolution_frame.grid(row=2, column=1, sticky=tk.W)
        for i in range(2):
            entry = tk.Entry(single_view_resolution_frame, textvariable=self.single_view_resolution[i],width=5,justify=tk.CENTER)
            entry.grid(row=0, column=i)
            single_view_resolution_entries.append(entry)
        ToolTip(single_view_resolution_label, self.config.__hints__["single_view_resolution"])

        # Save images checkbox
        self.save_images = BooleanVar(value=self.config.save_images)
        save_images_checkbox = tk.Checkbutton(video_config_frame, text="Save Images", variable=self.save_images)
        save_images_checkbox.grid(row=3, column=0, columnspan=2, sticky=tk.W)
        ToolTip(save_images_checkbox, self.config.__hints__["save_images"])

        # Save images path entry
        save_images_path_label = tk.Label(video_config_frame, text="Save Images Path")
        save_images_path_label.grid(row=4, column=0, sticky=tk.W)
        self.save_images_path = StringVar(value=self.config.save_images_path)
        save_images_path_entry = tk.Entry(video_config_frame, textvariable=self.save_images_path)
        save_images_path_entry.grid(row=4, column=1, columnspan=2, sticky=tk.W)
        ToolTip(save_images_path_label, self.config.__hints__["save_images_path"])

        # Save video path entry
        save_video_path_label = tk.Label(video_config_frame, text="Save Video Path")
        save_video_path_label.grid(row=5, column=0, sticky=tk.W)
        self.save_video_path = StringVar(value=self.config.save_video_path)
        save_video_path_entry = tk.Entry(video_config_frame, textvariable=self.save_video_path)
        save_video_path_entry.grid(row=5, column=1, columnspan=2, sticky=tk.W)
        ToolTip(save_video_path_label, self.config.__hints__["save_video_path"])

        # Video postfix time checkbox
        self.video_postfix_time = BooleanVar(value=self.config.video_postfix_time)
        video_postfix_time_checkbox = tk.Checkbutton(video_config_frame, text="Add Time Postfix to Video Name", variable=self.video_postfix_time)
        video_postfix_time_checkbox.grid(row=6, column=0, columnspan=3, sticky=tk.W)
        ToolTip(video_postfix_time_checkbox, self.config.__hints__["video_postfix_time"])

        # Visible camera views configuration
        camera_views_frame = tk.LabelFrame(dialog, text="Visible Camera Views", padx=5, pady=5)
        camera_views_frame.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        
        self.camera_views = {}
        for i in range(8):
            self.camera_views[i] = BooleanVar(value=self.config.visible_camera_views[i])
            if i not in available_cameras:
                self.camera_views[i].set(False)
            checkbox = tk.Checkbutton(camera_views_frame, text=f"Camera {i}" if i !=0 else "Camera Ego", variable=self.camera_views[i] )
            if i not in available_cameras:
                checkbox.config(state=tk.DISABLED)
            checkbox.grid(row=i//4, column=i%4, sticky=tk.W)
        #ToolTip(camera_views_frame, self.config.__hints__["visible_camera_views"])

        # Visualization configuration
        visualization_frame = tk.LabelFrame(dialog, text="Visualization Configuration", padx=5, pady=5)
        visualization_frame.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)

        depth_colormap_label = tk.Label(visualization_frame, text="Depth Colormap")
        depth_colormap_label.grid(row=0, column=0, sticky=tk.W)
        self.depth_colormap = StringVar(value=self.config.visualization_config["depth_colormap"])
        depth_colormap_combobox = ttk.Combobox(visualization_frame, textvariable=self.depth_colormap, values=["GRAY", "INFERNO", "OCEAN", "JET", "HOT"],width=8)
        depth_colormap_combobox.grid(row=0, column=1, sticky=tk.W)
        ToolTip(depth_colormap_label, self.config.__hints__["visualization_config"]["depth_colormap"])

        force_colormap_label = tk.Label(visualization_frame, text="Force Colormap")
        force_colormap_label.grid(row=1, column=0, sticky=tk.W)
        self.force_colormap = StringVar(value=self.config.visualization_config["force_color_map"])
        force_colormap_combobox = ttk.Combobox(visualization_frame, textvariable=self.force_colormap, values=["GRAY", "INFERNO", "OCEAN", "HOT"],width=8)
        force_colormap_combobox.grid(row=1, column=1, sticky=tk.W)
        ToolTip(force_colormap_label, self.config.__hints__["visualization_config"]["force_color_map"])

        mesh_color_label = tk.Label(visualization_frame, text="Mesh Color (R, G, B)")
        mesh_color_label.grid(row=2, column=0, sticky=tk.W)
        self.mesh_color = [IntVar(value=int(val * 255)) for val in self.config.visualization_config["mesh_color"]]  # Convert 0-1 to 0-255
        mesh_color_entries = []
        rgb_frame=tk.Frame(visualization_frame)
        rgb_frame.grid(row=2, column=1, sticky=tk.W)
        for i in range(3):
            entry = tk.Entry(rgb_frame, textvariable=self.mesh_color[i],width=8,justify=tk.CENTER)
            entry.grid(row=0, column=0+i, sticky=tk.W+tk.E)
            entry.bind("<KeyRelease>", lambda event, var=self.mesh_color[i]: enforce_int_limit(event, var))

            mesh_color_entries.append(entry)
        ToolTip(mesh_color_label, self.config.__hints__["visualization_config"]["mesh_color"])

        mesh_alpha_label = tk.Label(visualization_frame, text="Mesh Alpha")
        mesh_alpha_label.grid(row=3, column=0, sticky=tk.W)
        self.mesh_alpha = DoubleVar(value=self.config.visualization_config["mesh_alpha"])
        mesh_alpha_slider = tk.Scale(visualization_frame, variable=self.mesh_alpha, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL,showvalue=False)
        mesh_alpha_slider.grid(row=3, column=1, sticky=tk.W+tk.E, columnspan=1)
        mesh_alpha_entry = tk.Entry(visualization_frame, textvariable=self.mesh_alpha,width=8,justify=tk.CENTER)
        mesh_alpha_entry.grid(row=3, column=2, sticky=tk.W+tk.E )
        ToolTip(mesh_alpha_label, self.config.__hints__["visualization_config"]["mesh_alpha"])

        # Visibility configuration
        visibility_frame = tk.LabelFrame(dialog,text="Visibility Configuration", padx=5, pady=5)
        visibility_frame.grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)

      
        self.visibility_mode = StringVar(value="RGB" if self.config.visibility["RGB"] else "depth")
        rgb_radio = tk.Radiobutton(visibility_frame, text="RGB", variable=self.visibility_mode, value="RGB")
        rgb_radio.grid(row=0, column=0, sticky=tk.W)
        depth_radio = tk.Radiobutton(visibility_frame, text="Depth", variable=self.visibility_mode, value="depth")
        depth_radio.grid(row=0, column=1, sticky=tk.W)
        ToolTip(rgb_radio, self.config.__hints__["visibility"]["RGB"])
        ToolTip(depth_radio, self.config.__hints__["visibility"]["depth"])

        # Initialize visibility checkboxes dictionary
        self.visibility = {}
        for i, key in enumerate(self.config.visibility.keys()):
            if key not in ["RGB", "depth"]:
                self.visibility[key] = BooleanVar(value=self.config.visibility[key])
                checkbox = tk.Checkbutton(visibility_frame, text=key, variable=self.visibility[key])
                checkbox.grid(row=0+i//4, column=i%4, sticky=tk.W)
                ToolTip(checkbox, self.config.__hints__["visibility"][key])
        
        return 4

    def save(self):
        self.config.fps = self.fps.get()
        self.config.save_images = self.save_images.get()
        self.config.visible_camera_views = {i: self.camera_views[i].get() for i in self.camera_views}
        self.config.visualization_config["depth_colormap"] = self.depth_colormap.get()
        self.config.visualization_config["force_color_map"] = self.force_colormap.get()
        self.config.visualization_config["mesh_color"] = [c.get() / 255.0 for c in self.mesh_color]  # Convert 0-255 to 0-1
        self.config.visualization_config["mesh_alpha"] = self.mesh_alpha.get()

        # Update visibility based on radio button selection
        self.config.visibility["RGB"] = self.visibility_mode.get() == "RGB"
        self.config.visibility["depth"] = self.visibility_mode.get() == "depth"
        for key in self.visibility:
            self.config.visibility[key] = self.visibility[key].get()

        self.config.frame_range = [fr.get() for fr in self.frame_range]
        self.config.single_view_resolution = [res.get() for res in self.single_view_resolution]
        self.config.save_images_path = self.save_images_path.get()
        self.config.save_video_path = self.save_video_path.get()
        self.config.video_postfix_time = self.video_postfix_time.get()


