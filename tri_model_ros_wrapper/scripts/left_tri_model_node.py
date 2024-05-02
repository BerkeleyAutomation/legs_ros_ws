#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import torch
import torch.nn.functional as tfn
import os
import numpy as np
torch._C._jit_set_profiling_executor(False)

def is_tensor(data):
    return type(data) == torch.Tensor

def is_tuple(data):
    return isinstance(data, tuple)

def is_list(data):
    return isinstance(data, list) or isinstance(data, torch.nn.ModuleList)

def is_dict(data):
    return isinstance(data, dict) or isinstance(data, torch.nn.ModuleDict)

def is_seq(data):
    return is_tuple(data) or is_list(data)

def iterate1(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var, *args, **kwargs):
        if is_seq(var):
            return [func(v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(var, *args, **kwargs)
    return inner


@iterate1
def interpolate(tensor, size, scale_factor, mode):
    if size is None and scale_factor is None:
        return tensor
    if is_tensor(size):
        size = size.shape[-2:]
    return tfn.interpolate(
        tensor, size=size, scale_factor=scale_factor,
        recompute_scale_factor=False, mode=mode,
        align_corners=None,
    )


def resize_input(
    rgb: torch.Tensor,
    intrinsics: torch.Tensor = None,
    resize: tuple = None
):
    """Resizes input data

    Args:
        rgb (torch.Tensor): input image (B,3,H,W)
        intrinsics (torch.Tensor): camera intrinsics (B,3,3)
        resize (tuple, optional): resize shape. Defaults to None.

    Returns:
        rgb: resized image (B,3,h,w)
        intrinsics: resized intrinsics (B,3,3)
    """
    # Don't resize if not requested
    if resize is None:
        if intrinsics is None:
            return rgb
        else:
            return rgb, intrinsics
    # Resize rgb
    orig_shape = [float(v) for v in rgb.shape[-2:]]
    rgb = interpolate(rgb, mode="bilinear", scale_factor=None, size=resize)
    # Return only rgb if there are no intrinsics
    if intrinsics is None:
        return rgb
    # Resize intrinsics
    shape = [float(v) for v in rgb.shape[-2:]]
    intrinsics = intrinsics.clone()
    intrinsics[:, 0] *= shape[1] / orig_shape[1]
    intrinsics[:, 1] *= shape[0] / orig_shape[0]
    # return resized input
    return rgb, intrinsics

class StereoModel(torch.nn.Module):
    """Learned Stereo model.

    Takes as input two images plus intrinsics and outputs a metrically scaled depth map.

    Taken from: https://github.com/ToyotaResearchInstitute/mmt_stereo_inference
    Paper here: https://arxiv.org/pdf/2109.11644.pdf
    Authors: Krishna Shankar, Mark Tjersland, Jeremy Ma, Kevin Stone, Max Bajracharya

    Pre-trained checkpoint here: s3://tri-ml-models/efm/depth/stereo.pt

    Args:
        cfg (Config): configuration file to initialize the model
        ckpt (str, optional): checkpoint path to load a pre-trained model. Defaults to None.
        baseline (float): Camera baseline. Defaults to 0.12 (ZED baseline)
    """

    def __init__(self, ckpt: str = None):
        super().__init__()
        # Initialize model
        self.model = torch.jit.load(ckpt).cuda()
        self.model.eval()

    def inference(
        self,
        rgb_left: torch.Tensor,
        rgb_right: torch.Tensor,
        intrinsics: torch.Tensor,
        resize: tuple = None,
        baseline: float = 0.12
    ):
        """Performs inference on input data

        Args:
            rgb_left (torch.Tensor): input float32 image (B,3,H,W)
            rgb_right (torch.Tensor): input float32 image (B,3,H,W)
            intrinsics (torch.Tensor): camera intrinsics (B,3,3)
            resize (tuple, optional): resize shape. Defaults to None.

        Returns:
            depth: output depth map (B,1,H,W)
        """
        rgb_left, intrinsics = resize_input(
            rgb=rgb_left, intrinsics=intrinsics, resize=resize
        )
        rgb_right = resize_input(rgb=rgb_right, resize=resize)

        with torch.no_grad():
            output, _ = self.model(rgb_left, rgb_right)

        disparity_sparse = output["disparity_sparse"]
        mask = disparity_sparse != 0
        depth = torch.zeros_like(disparity_sparse)
        # depth[mask] = baseline * intrinsics[0, 0, 0] / disparity_sparse[mask]
        depth = baseline * intrinsics[0, 0, 0] / output["disparity"]
        rgb = (rgb_left.squeeze(0).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)
        return depth, output["disparity"], disparity_sparse,rgb

def format_image(rgb):
    return torch.tensor(rgb.transpose(2,0,1)[None]).to(torch.float32).cuda() / 255.0

def deproject_to_RGB_point_cloud(image, depth_image, camera, num_samples = 844800//2, device = 'cuda'):
        """
        Converts a depth image into a point cloud in world space using a Camera object.
        """
        scale = 1
        # import pdb; pdb.set_trace()
        # H = self.pipeline.datamanager.train_dataparser_outputs.dataparser_transform
        # c2w = camera.camera_to_worlds.cpu()
        # depth_image = depth_image.cpu()
        # image = image.cpu()
        # c2w = camera.camera_to_worlds.to(device)
        depth_image = depth_image.to(device)
        image = image.to(device)
        fx = camera[0,0,0]
        fy = camera[0,1,1]
        # cx = camera.cx.item()
        # cy = camera.cy.item()

        _, _, height, width = depth_image.shape

        grid_x, grid_y = torch.meshgrid(torch.arange(width, device = device), torch.arange(height, device = device), indexing='ij')
        grid_x = grid_x.transpose(0,1).float()
        grid_y = grid_y.transpose(0,1).float()

        flat_grid_x = grid_x.reshape(-1)
        flat_grid_y = grid_y.reshape(-1)
        flat_depth = depth_image[0, 0].reshape(-1)
        flat_image = image.reshape(-1, 3)

        ### simple uniform sampling approach
        # num_points = flat_depth.shape[0]
        # sampled_indices = torch.randint(0, num_points, (num_samples,))
        non_zero_depth_indices = torch.nonzero(flat_depth != 0).squeeze()

        # Ensure there are enough non-zero depth indices to sample from
        if num_samples is not None:
            if non_zero_depth_indices.numel() < num_samples:
                num_samples = non_zero_depth_indices.numel()
            # Sample from non-zero depth indices
            sampled_indices = non_zero_depth_indices[torch.randint(0, non_zero_depth_indices.shape[0], (num_samples,))]
        else:
            sampled_indices = non_zero_depth_indices

        sampled_depth = flat_depth[sampled_indices] * scale
        # sampled_depth = flat_depth[sampled_indices]
        sampled_grid_x = flat_grid_x[sampled_indices]
        sampled_grid_y = flat_grid_y[sampled_indices]
        sampled_image = flat_image[sampled_indices]

        X_camera = (sampled_grid_x - width/2) * sampled_depth / fx
        Y_camera = -(sampled_grid_y - height/2) * sampled_depth / fy

        ones = torch.ones_like(sampled_depth)
        P_camera = torch.stack([X_camera, Y_camera, sampled_depth, ones], dim=1)
        
        # homogenizing_row = torch.tensor([[0, 0, 0, 1]], dtype=c2w.dtype, device=device)
        # camera_to_world_homogenized = torch.cat((c2w, homogenizing_row), dim=0)

        # P_world = torch.matmul(camera_to_world_homogenized, P_camera.T).T
        
        return P_camera[:, :3], sampled_image

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        stereo_ckpt = "./stereo_20230724.pt"
        # filepath = "./AUTOLAB_Fri_Jul__7_14_57_48_2023"
        file_path = os.path.abspath(__file__)
        folder_path = file_path[:file_path.rfind('/')]
        stereo_ckpt = folder_path + '/../../../../src/tri_model_ros_wrapper/models/stereo_20230724.pt'
        self.model_ = StereoModel(stereo_ckpt)
        self.model_.cuda()
        self.cv_bridge = CvBridge()
        self.left_zed_left_sub = Subscriber(
            self,
            Image,
            '/left_zed/zed_node/left/image_rect_color'
        )
        self.left_zed_right_sub = Subscriber(
            self,
            Image,
            '/left_zed/zed_node/right/image_rect_color'
        )
        self.left_rgb_pub = self.create_publisher(Image,'tri_left_zed_cropped',10)
        self.left_depth_pub = self.create_publisher(Image,'tri_left_zed_depth',10)
        self.left_camera_info_sub = self.create_subscription(CameraInfo,'/left_zed/zed_node/left/camera_info',self.leftCameraInfoCallback,10)
        self.left_intrinsics_ = None
        self.left_synchronizer = ApproximateTimeSynchronizer(
            [self.left_zed_left_sub, self.left_zed_right_sub],
            queue_size=10,
            slop=0.1
        )
        self.left_synchronizer.registerCallback(self.leftCallback)

    def leftCameraInfoCallback(self,msg):
        fx,_,cx,_,fy,cy,_,_,_ = msg.k
        self.left_intrinsics_ = torch.tensor([[
            [fx,0,cx],
            [0,fy,cy],
            [0,0,1]
        ]]).to(torch.float32).cuda()

    def rightCameraInfoCallback(self,msg):
        fx,_,cx,_,fy,cy,_,_,_ = msg.k
        self.right_intrinsics_ = torch.tensor([[
            [fx,0,cx],
            [0,fy,cy],
            [0,0,1]
        ]]).to(torch.float32).cuda()

    def leftCallback(self, left_msg, right_msg):
        if(self.left_intrinsics_ is None):
            return None
        left_ = self.cv_bridge.imgmsg_to_cv2(left_msg)
        right_ = self.cv_bridge.imgmsg_to_cv2(right_msg)
        height, width, _ = left_.shape
        height = height - height % 32
        width = width - width % 32
        left_ = left_[:height, :width, :3]
        right_ = right_[:height, :width, :3]
        tridepth, disparity, disparity_sparse,cropped_rgb = self.model_.inference(
            rgb_left=format_image(left_),
            rgb_right=format_image(right_),
            intrinsics=self.left_intrinsics_,
            resize=left_.shape[:2],
            baseline=0.12
        )
        depth_np = tridepth.squeeze(0).squeeze(0).cpu().detach().numpy()
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_np)
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(cropped_rgb)
        depth_msg.header.stamp = left_msg.header.stamp
        rgb_msg.header.stamp = rgb_msg.header.stamp
        self.left_depth_pub.publish(depth_msg)
        self.left_rgb_pub.publish(rgb_msg)
        print("Published Left",flush=True)

    def rightCallback(self, left_msg, right_msg):
        if(self.right_intrinsics_ is None):
            return None
        left_ = self.cv_bridge.imgmsg_to_cv2(left_msg)
        right_ = self.cv_bridge.imgmsg_to_cv2(right_msg)
        height, width, _ = left_.shape
        height = height - height % 32
        width = width - width % 32
        left_ = left_[:height, :width, :3]
        right_ = right_[:height, :width, :3]
        tridepth, disparity, disparity_sparse,cropped_rgb = self.model_.inference(
            rgb_left=format_image(left_),
            rgb_right=format_image(right_),
            intrinsics=self.right_intrinsics_,
            resize=left_.shape[:2],
            baseline=0.12
        )
        depth_np = tridepth.squeeze(0).squeeze(0).cpu().detach().numpy()
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_np)
        rgb_msg = self.cv_bridge.cv2_to_compressed_imgmsg(cropped_rgb)
        depth_msg.header.stamp = left_msg.header.stamp
        rgb_msg.header.stamp = left_msg.header.stamp
        self.right_depth_pub.publish(depth_msg)
        self.right_rgb_pub.publish(rgb_msg)
        print("Published Right",flush=True)
       
def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()