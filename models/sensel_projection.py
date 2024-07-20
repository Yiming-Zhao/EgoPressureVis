import numpy as np
import cv2

sensel_w = 0.240
sensel_h = 0.1375
sensel_origin_x = 0.0
sensel_origin_y = 0.0
sensel_z = 0.0
sensel_corners_3D = np.array([[sensel_origin_x + sensel_w/2, sensel_origin_y - sensel_h/2, sensel_z],
                              [sensel_origin_x + sensel_w/2, sensel_origin_y + sensel_h/2, sensel_z],
                              [sensel_origin_x - sensel_w/2, sensel_origin_y + sensel_h/2, sensel_z], 
                              [sensel_origin_x - sensel_w/2, sensel_origin_y - sensel_h/2, sensel_z]],dtype=np.float32)
sensel_pad_2D = np.array([[185, 0], [185, 105], [0, 105], [0, 0]],dtype=np.float32)



def project_rectangle(rectangle_3d, K,R,t, scale_factor=1.0):
 
    # Convert the 4 coordinates of the rectangle area to a numpy array
    rectangle_3d = np.array(rectangle_3d)
    original_rectangle_3d = rectangle_3d.copy()
    if  scale_factor != 1.0:
        
        # Calculate the center of the rectangle area
        rectangle_3d_center = np.mean(rectangle_3d, axis=0)

        # Translate the rectangle area so that its center is at the origin
        rectangle_3d -= rectangle_3d_center

        # Calculate the width and height of the rectangle area
        width = np.linalg.norm(rectangle_3d[1] - rectangle_3d[0])
        height = np.linalg.norm(rectangle_3d[3] - rectangle_3d[0])

        # Calculate the aspect ratio of the rectangle area
    
        aspect_ratio = width / height
    
        # Enlarge the rectangle area by the given scale factor while keeping the aspect ratio
        rectangle_3d *= scale_factor / aspect_ratio if aspect_ratio > 1 else scale_factor * aspect_ratio
        rectangle_3d += rectangle_3d_center

    # Extract camera parameters from the dictionary
   
    #image is already undistorted
    dist_coeffs=np.array([])

    rvec, jacobian = cv2.Rodrigues(R)
    tvec = t # tvec = t/ 1000.0  # Convert to meters
    # Project the 3D coordinates of the rectangle area onto the 2D image plane of the camera
    rectangle_2d, _ = cv2.projectPoints(rectangle_3d, rvec, tvec, K, dist_coeffs)
    original_rectangle_2d,_ = cv2.projectPoints(original_rectangle_3d, rvec, tvec, K, dist_coeffs)
    homography, status = cv2.findHomography(sensel_pad_2D, original_rectangle_2d[:, 0, :])
    # The projected rectangle_2d contains the pixel coordinates of the rectangle area in the image plane of the camera
    return rectangle_2d, homography