

import torch
import numpy as np
import pytorch3d
import pytorch3d.renderer
import cv2
from scipy.spatial.transform import Rotation
from manopth.manolayer import ManoLayer
from models.io import mano_ncomps,mano_root
from models.io import colormaps
def get_3d_joints_in_camera_frame(vertices, camera_rot,camera_trans):
    '''
    get 3D joints in the camera frame
    '''
    points = torch.einsum('bij,bkj->bki', camera_rot, vertices)
    camera_trans=camera_trans.view(-1,1,3)

    points = points + camera_trans
    return points

def get_2d_points(points,camera_intrinsic,image_size):
    '''
    get 2D vertices in the image plane in Kaolin renderer's coordinate system
    '''
    projected_points = points / points[:,:,-1].unsqueeze(-1)
    height,width=image_size
    B, N, _ = points.shape
    intrinsics=camera_intrinsic.clone().float().unsqueeze(0).repeat(B,1,1)
    projected_points = torch.einsum('bij,bkj->bki', intrinsics, projected_points)
    projected_points[:,:,0]=projected_points[:,:,0]
    return projected_points[:, :, :-1]

def draw_skeleton(image, joints, thickness=2):
    ''' Draw the skeleton on the image
    image: (H, W, 3), the image
    joints: (N, 2), the joints in the image coordinates
    thickness: int, the thickness of skeleton
    :return: the image with skeleton
    '''
    image = image.copy()
    thumb = [0, 1, 2, 3, 4] 
    thumb_color = (0, 255, 0)
    index = [0, 5, 6, 7,8]
    index_color = (0, 0, 255)
    middle = [0, 9, 10, 11,12]
    middle_color = (255, 0, 0)
    ring = [0, 13, 14, 15,16]
    ring_color = (255, 255, 0)
    pinky = [0, 17, 18, 19,20]
    pinky_color = (255, 0, 255)

    for i in range(0,len(thumb)-1):
        jidx=thumb[i]
        jidx_next=thumb[i+1]
        cv2.line(image, tuple(joints[jidx, :2].astype(int)), tuple(joints[jidx_next, :2].astype(int)), thumb_color, thickness)
        cv2.circle(image, tuple(joints[jidx_next, :2].astype(int)), int(2*thickness), thumb_color, -1)
    for i in range(0,len(index)-1):
        jidx=index[i]
        jidx_next=index[i+1]
        cv2.line(image, tuple(joints[jidx, :2].astype(int)), tuple(joints[jidx_next, :2].astype(int)), index_color, thickness)
        cv2.circle(image, tuple(joints[jidx_next, :2].astype(int)), int(2*thickness), index_color, -1)
    for i in range(0,len(middle)-1):
        jidx=middle[i]
        jidx_next=middle[i+1]
        cv2.line(image, tuple(joints[jidx, :2].astype(int)), tuple(joints[jidx_next, :2].astype(int)), middle_color, thickness)
        cv2.circle(image, tuple(joints[jidx_next, :2].astype(int)), int(2*thickness), middle_color, -1)    
    for i in range(0,len(ring)-1):
        jidx=ring[i]
        jidx_next=ring[i+1]
        cv2.line(image, tuple(joints[jidx, :2].astype(int)), tuple(joints[jidx_next, :2].astype(int)), ring_color, thickness)
        cv2.circle(image, tuple(joints[jidx_next, :2].astype(int)), int(2*thickness), ring_color, -1)  
    for i in range(0,len(pinky)-1):
        jidx=pinky[i]
        jidx_next=pinky[i+1]
        cv2.line(image, tuple(joints[jidx, :2].astype(int)), tuple(joints[jidx_next, :2].astype(int)), pinky_color, thickness)
        cv2.circle(image, tuple(joints[jidx_next, :2].astype(int)), int(2*thickness), pinky_color, -1) 
    cv2.circle(image, tuple(joints[0, :2].astype(int)), int(2*thickness), (255, 255, 255), -1)


    return image



def render_mesh(vertices, faces, translation, focal_length, height, width, principal_point=((0.0, 0.0),), device=None,color=(0.4882353,  0.3117647,0.25098039  )
):
    ''' Render the mesh under camera coordinates
    vertices: (N_v, 3), vertices of mesh
    faces: (N_f, 3), faces of mesh
    translation: (3, ), translations of mesh or camera
    focal_length: float, focal length of camera
    height: int, height of image
    width: int, width of image
    device: "cpu"/"cuda:0", device of torch
    :return: the rgba rendered image
    '''
    if device is None:
        device = vertices.device

    bs = vertices.shape[0]

    # add the translation
    if translation is not None:
        vertices = vertices + translation[:, None, :]

    # upside down the mesh
    # rot = Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix().astype(np.float32)
    
    rot = Rotation.from_euler('z', 180, degrees=True).as_matrix().astype(np.float32)
    rot = torch.from_numpy(rot).to(device).expand(bs, 3, 3)
    vertices = torch.matmul(rot, vertices.transpose(1, 2)).transpose(1, 2)
    faces = faces.expand(bs, *faces.shape).to(device)

    

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(vertices)  # (B, V, 3)
    verts_rgb[:,:,0]=color[0]
    verts_rgb[:,:,1]=color[1]
    verts_rgb[:,:,2]=color[2]
    
    textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)
    mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures)

    # Initialize a camera.
    #rr=axis_angle_to_matrix(torch.tensor([np.pi,0,0])).repeat(vertices.shape[0],1,1).float().to(vertices.device)
    cameras = pytorch3d.renderer.PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        device=device,
        in_ndc=False,
        image_size=((height, width),),
     #   R=rr
    )

    # Define the settings for rasterization and shading.
    raster_settings = pytorch3d.renderer.RasterizationSettings(
        image_size=(height, width),   # (H, W)
        # image_size=height,   # (H, W)
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Define the material
    materials = pytorch3d.renderer.Materials(
        ambient_color=((1, 1, 1),),
        diffuse_color=((1, 1, 1),),
        specular_color=((1, 1, 1),),
        shininess=32,
        device=device
    )

    # Place a directional light in front of the object.
    lights = pytorch3d.renderer.DirectionalLights(device=device, direction=((0, 0, -1),))

    # Create a phong renderer by composing a rasterizer and a shader.
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=pytorch3d.renderer.SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials
        )
    )

    # Do rendering
    imgs = renderer(mesh)
    return imgs
def render_obj(image,verts_batch,transl_batch,focal,principal_point,faces,alpha=0.5):
        color_batch = render_mesh(
        vertices=verts_batch, faces=faces,
        translation=transl_batch,
        principal_point=principal_point,
        focal_length=focal, height=image.shape[-3], width=image.shape[-2])
        

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch
        valid_mask = valid_mask_batch.repeat(1,1,1,3).cpu().numpy()
        if image.dim() == 3:
            input_img = (image*255.0).repeat(verts_batch.shape[0],1,1,1).cpu().numpy()
        else:
            input_img = (image*255.0).cpu().numpy()
        
        image_vis = alpha * color[:,:, :, :3] * valid_mask + (
        1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img
        image_vis=image_vis[:,:,:,[2,1,0]]
      #  image_vis = image_vis.astype(np.uint8)
      #  image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
        
        color=color[:,:,:,[2,1,0]]
        color =  color[:,:, :, :3] * valid_mask + (1 - valid_mask) * 255.0
        valid_mask=valid_mask[...,0]*255.0
        
  
        return {"rgb":image_vis,"foreground":color,"mask":valid_mask}

def render_obj_foreground(image_shape,verts_batch,transl_batch,focal,principal_point,faces,color=(0.4882353,  0.3117647,0.25098039 )):
        color_batch = render_mesh(
        vertices=verts_batch, faces=faces,
        translation=transl_batch,
        principal_point=principal_point,
        focal_length=focal, height=image_shape[0], width=image_shape[1],color=color)
        

        valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        image_vis_batch = (image_vis_batch * 255).cpu().numpy()

        color = image_vis_batch
        valid_mask = valid_mask_batch.repeat(1,1,1,3).cpu().numpy()

        color=color[:,:,:,[2,1,0]]
        color =  color[:,:, :, :3] * valid_mask + (1 - valid_mask) * 255.0
        valid_mask=valid_mask[...,0]*255.0
        
  
        return {"foreground":color,"mask":valid_mask}
   


class ManoObject(torch.nn.Module):
    def __init__(self,side,device="cuda:0",mano_root=mano_root,mano_ncomps=mano_ncomps):
        super(ManoObject,self).__init__()

        self.mano_mesh=ManoLayer(flat_hand_mean=True,
                            side=side,
                            mano_root=mano_root,
                            ncomps= mano_ncomps,
                            use_pca=False,
                            root_rot_mode="axisang",
                            joint_rot_mode="axisang",
                            center_idx=None)
        self.mano_mesh.to(device)
        self.mano_mesh.train(False) 
        self.side=side

    def get_3d_joints(self, vertices):

        joints = torch.einsum('bik,ji->bjk', [vertices, self.mano_mesh.th_J_regressor])
       
        tips = vertices[:, [729, 319, 445, 556, 673]] # Could be different

        joints = torch.cat([joints, tips], 1)
        joints=joints[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
        return joints
    
    def forward(self,thetas,transl,betas,displacement=None):
        batch_size=len(thetas)
        thetas=thetas.view([batch_size,-1])
        #print(global_orientation,joint_rotation)
        
        mano_outputs=self.mano_mesh( thetas,
                th_betas=betas,
                th_trans=None)
        v, jtr=mano_outputs[0],mano_outputs[1]
        
        if displacement is not None:
            disp=displacement["displacement"]
            normals=displacement["normals"]
            if normals is not None:
                v=v+disp*normals
            else:
                v=v+disp
        v=v/1000  
        jtr=jtr/1000
        v= v +  transl.view(-1, 1, 3)
        jtr=jtr+transl.view(-1, 1, 3)

     
        return v,jtr

def apply_colormap_on_depth_image(depth_image,color_map="GRAY"):
        

        # Calculate the min and max values
        min_val = np.min(depth_image)
        max_val = np.max(depth_image)

        # ormalize the image to the range [0, 255]
        depth_image_normalized = (depth_image - min_val) / (max_val - min_val) * 255

        depth_image_normalized_8bit = depth_image_normalized.astype(np.uint8)

        

        cmap=colormaps[color_map]
        colored_depth = cv2.applyColorMap(depth_image_normalized_8bit, cmap)
        
        return colored_depth, min_val, max_val
