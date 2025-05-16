from nuscenes import NuScenes
import cv2
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
import sys
import json
import os
import shutil
sys.path.append('/home//UniAD')
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
from PIL import Image, ImageDraw, ImageFont
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import BoxVisibility
from PIL import ImageDraw, ImageFont
import random
from nuscenes.utils.splits import create_splits_scenes
import pickle
from typing import List, Tuple, Union
from shapely.geometry import MultiPoint, box
import copy
def get_ego_speed_from_sample(nusc, sample_token):
    """
    sample tokenLiDAR
    
    :
        nusc: NuScenes
        sample_token: sampletoken
        
    :
        speed: m/s
    """
    # sample
    sample = nusc.get('sample', sample_token)
    
    # LiDAR
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    
    # LiDARego_pose
    current_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    current_position = np.array(current_ego_pose['translation'])
    current_timestamp = current_ego_pose['timestamp'] / 1000000  # 
    
    # LiDAR
    prev_lidar_token = lidar_data['prev']
    if not prev_lidar_token:
        # LiDARsample
        prev_sample_token = sample['prev']
        if not prev_sample_token:
            print("LiDAR")
            return 0.0
        
        # sampleLiDAR
        prev_sample = nusc.get('sample', prev_sample_token)
        prev_lidar_token = prev_sample['data']['LIDAR_TOP']
    
    # LiDAR
    prev_lidar_data = nusc.get('sample_data', prev_lidar_token)
    prev_ego_pose = nusc.get('ego_pose', prev_lidar_data['ego_pose_token'])
    prev_position = np.array(prev_ego_pose['translation'])
    prev_timestamp = prev_ego_pose['timestamp'] / 1000000  # 
    
    #  ()
    dt = current_timestamp - prev_timestamp
    if dt <= 0:
        print("")
        return 0.0
    
    #  ()
    displacement = current_position - prev_position
    
    #  (z)
    distance = np.sqrt(displacement[0]**2 + displacement[1]**2)
    
    #  (/)
    speed = distance / dt
    
    return speed
def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None
def get_2d_center_in_camera(annotation_token, nusc,camera_name='CAM_FRONT'):
    """
    Given the annotation token, this function calculates the 2D projection of the object center in the 'CAM_FRONT' view.
    
    :param annotation_token: Token for a sample annotation.
    :param nusc: The NuScenes dataset object.
    
    :return: The 2D (x, y) coordinates of the object center in the 'CAM_FRONT' view.
    """
    # Retrieve the annotation record
    ann_rec = nusc.get('sample_annotation', annotation_token)
    
    # Get the token for the associated sample_data (we are interested in the 'CAM_FRONT' view)
    sample_token = ann_rec['sample_token']
    sample_data = nusc.get('sample', sample_token)['data']

    
    cam_front_token = sample_data[camera_name]
    
    # Get camera intrinsic parameters for 'CAM_FRONT'
    sd_rec = nusc.get('sample_data', cam_front_token)
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
    
    # Get the annotation box (3D bounding box)
    box = nusc.get_box(ann_rec['token'])
    translation = ann_rec['translation']
    # Retrieve ego pose (translation and rotation)
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    translation_ego = np.array(pose_rec['translation'])
    rotation_ego = Quaternion(pose_rec['rotation'])
    
    # Retrieve calibrated sensor pose (translation and rotation)
    translation_sensor = np.array(cs_rec['translation'])
    rotation_sensor = Quaternion(cs_rec['rotation'])
    
    # Step 1: Transform the 3D box center from the world (ego) coordinate system to the sensor coordinate system.
    box.translate(-translation_ego)  # Translate by ego translation
    box.rotate(rotation_ego.inverse)  # Rotate by inverse ego rotation
    
    box.translate(-translation_sensor)  # Translate by sensor translation
    box.rotate(rotation_sensor.inverse)  # Rotate by inverse sensor rotation
    
    # Step 2: Get the 3D corners of the box and take the center point
    corners_3d = box.corners()  # 3D corners of the bounding box
    in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
    corners_3d = corners_3d[:, in_front]
    
    center_3d = box.center  # The center of the 3D bounding box
    
    # Step 3: Project the 3D center to 2D using camera intrinsic parameters
    corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()  # Project 3D corners to 2D
    final_coords = post_process_coords(corner_coords)
    # center_2d = view_points(np.array([[center_3d[0]], [center_3d[1]], [center_3d[2]]]), camera_intrinsic, True).T[0]
    #corner_coords
    if final_coords is None:
        return None, None, None, None
    else:
        x_min,y_min , x_max, y_max = final_coords
    # x_min = min(1600,max(0, int(np.min(corner_coords[:, 0]))))
    # x_max = max(0,min(1600, int(np.max(corner_coords[:, 0]))))
    # y_min = min(900,max(0, int(np.min(corner_coords[:, 1]))))
    # y_max = max(0,min(900, int(np.max(corner_coords[:, 1]))))
    # 
    x_box, y_box= (x_min + x_max) // 2, (y_min + y_max) // 2
    height = y_max - y_min
    width = x_max - x_min
    # if (int(height)==900 and int(width)==1600) or int(height)==0 or int(width)==0:
    #     height=0
    #     width=0
    # if x_box==1600 or y_box==900 or x_box==0 or y_box==0 or x_box==800 or y_box==450:
    #     #
    #     for i in range(8):
    #         x,y=corner_coords[i]
    #         if x<0 or x>1600 or y<0 or y>900:
    #             x_box,y_box=10000,10000
            
    # 
    # #
    # if x_box==10000 and y_box==10000:
    #     x,y=10000,10000
    # else:
    # x, y = int(center_2d[0]), int(center_2d[1])
    

    #
    
    return (int(x_box),int(y_box),int(height),int(width))
def obtain_map_info(nusc,
                    nusc_maps,
                    sample,
                    patch_size=(102.4, 102.4),
                    canvas_size=(256, 256),
                    layer_names=['lane_divider', 'road_divider'],
                    thickness=10):
    """
    Export 2d annotation from the info file and raw data.
    """
    l2e_r = sample['lidar2ego_rotation']
    l2e_t = sample['lidar2ego_translation']
    e2g_r = sample['ego2global_rotation']
    e2g_t = sample['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    nusc_map = nusc_maps[log['location']]
    if layer_names is None:
        layer_names = nusc_map.non_geometric_layers

    l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
    l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
    patch_box = (l2g_t[0], l2g_t[1], patch_size[0], patch_size[1])
    patch_angle = math.degrees(Quaternion(matrix=l2g_r_mat).yaw_pitch_roll[0])

    map_mask = nusc_map.get_map_mask(
        patch_box, patch_angle, layer_names, canvas_size=canvas_size)
    map_mask_road=map_mask[0]
    # for i in range(len(map_mask)-1):
    #     map_mask_middle=map_mask_middle|map_mask[i+1]

    map_mask_road = map_mask_road
    map_mask_road = map_mask_road[np.newaxis, :]
    map_mask_road = map_mask_road.transpose((2, 1, 0)).squeeze(2)  # (H, W, C)

    map_mask_lane = map_mask[1]
    map_mask_lane = map_mask_lane[np.newaxis, :]
    map_mask_lane = map_mask_lane.transpose((2, 1, 0)).squeeze(2)  # (H, W, C)


    erode = nusc_map.get_map_mask(patch_box, patch_angle, [
                                  'drivable_area'], canvas_size=canvas_size)
    erode = erode.transpose((2, 1, 0)).squeeze(2)

    map_mask = np.concatenate([erode[None], map_mask_road[None], map_mask_lane[None]], axis=0)
    return map_mask
class BEVRender():
    def __init__(self,
                 figsize=(20, 20),
                 margin: float = 25,
                 view: np.ndarray = np.eye(4),
                 show_gt_boxes=False):
        self.figsize = figsize
        self.fig, self.axes = None, None
        self.margin = margin
        self.view = view
        self.show_gt_boxes = show_gt_boxes
        #xy
       

    # def render_ego_vehicle(self, nusc, sample_token):
    # # 
    #     sample_record = nusc.get('sample', sample_token)
    #     # LIDAR_TOP
    #     lidar_record = sample_record['data']['LIDAR_TOP']
    #     # 
    #     sd_rec = nusc.get('sample_data', lidar_record)
    #     pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        
    #     # 
    #     ego_translation = np.array(pose_record['translation'])  #  (x, y, z)
    #     ego_rotation = Quaternion(pose_record['rotation'])  #  ()
        
    #     # 2.04.0
    #     width, length = 2.0, 4.0
    #     car_center = ego_translation[:2]  # x, y
    #     car_angle = ego_rotation.yaw_pitch_roll[0]  # yaw
        
    #     # 
    #     # Box(center, size, orientation), size=(length, width, height)height1.5
    #     ego_box = Box(
    #         center=ego_translation,  # 
    #         size=(length, width, 1.5),  # 
    #         orientation=ego_rotation  # 
    #     )
        
    #     # 
    #     c = np.array([1.0, 0.0, 0.0])  # 
    #     ego_box.render(self.axes, view=self.view, colors=(c, c, c))
    def save_future_positions_to_json(self, future_ego_xy_local, future_vehilces):
        """
         ego vehicle  JSON 
        """
        # 
        data = {}
        
        #  ego vehicle 
        data["ego vehicle"] = future_ego_xy_local.tolist()

        # 
        for count_car, future_xy in enumerate(future_vehilces, start=1):
            key = f"vehicle{count_car}"
            data[key] = future_xy.tolist()

        #  JSON 
        with open('/home//VLM_attack_propose/future_positions.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        print(" future_positions.json ")
    def render_sdc_car(self):
        sdc_car_png = cv2.imread('/home//VLM_attack_propose/sources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))
        #xy
        c = np.array([0, 0.8, 0])
        # self.axes.annotate('', xy=(10, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=c, mutation_scale=50))
        self.axes.annotate('', xy=(0, 2), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=c, mutation_scale=60,linewidth=2))
        # self.axes.text(10, 0, 'x', color='black', fontsize=40)
        # self.axes.text(1, 10, 'y', color='black', fontsize=40)

    def render_anno_data(
        self,
        sample_token,
        nusc,
        predict_helper):
        """
         JSON 
        """
        # 
        sample_record = nusc.get('sample', sample_token)
        
        # 
        future_ego_xy_local = predict_helper.get_future_for_ego_vehicle(sample_token, seconds=3, in_agent_frame=True)
        future_ego_xy_local = np.concatenate([[np.array([0.0, 0.0])], future_ego_xy_local], axis=0)
        c = np.array([0, 0.8, 0])
        # self._render_traj(future_ego_xy_local, line_color=c, dot_color=(0, 0, 0))

        #  LIDAR_TOP
        assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'
        
        #  LIDAR_TOP 
        lidar_record = sample_record['data']['LIDAR_TOP']
        
        #  LIDAR_TOP 
        data_path, boxes, _ = nusc.get_sample_data(lidar_record, selected_anntokens=sample_record['anns'])
        
        # 
        future_vehicles = []

        # 
        count_car = 0
        needToRenderNumber = False
        lidar_sample_data = nuscenes.get('sample_data',nuscenes.get('sample', sample_token)['data']['LIDAR_TOP'])
        sd_ep = nuscenes.get("ego_pose", lidar_sample_data["ego_pose_token"])
        ego_translation = sd_ep['translation']
        
        boxes = sorted(boxes, key=lambda box: np.max(np.linalg.norm(np.array(nusc.get('sample_annotation', box.token)['translation']) - np.array(ego_translation))))
        for box in boxes:
            instance_token = nusc.get('sample_annotation', box.token)['instance_token']
            instance = nusc.get('instance', instance_token)
            category_token = instance['category_token']
            category = nusc.get('category', category_token)

            # 
            if category['name'].split('.')[0] == 'vehicle' and category['name'].split('.')[1]!='bicycle' and category['name'].split('.')[1]!='motorcycle':
                needToRenderNumber = True
                count_car += 1
            else:
                needToRenderNumber = False
            
            future_xy_local = predict_helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=True)
            future_xy_global = predict_helper.get_future_for_agent(instance_token, sample_token, seconds=3, in_agent_frame=False)
            
            if future_xy_local.shape[0] > 0:
                trans = box.center
                rot = Quaternion(matrix=box.rotation_matrix)
                future_xy = convert_local_coords_to_global(future_xy_local, trans, rot)
                future_xy = np.concatenate([trans[None, :2], future_xy], axis=0)
                
                c = np.array([0, 0.8, 0])
                c_human = np.array([0, 0, 1])
                c_barrier = np.array([1, 0, 0])
                if needToRenderNumber:
                    box.render(self.axes, view=self.view, colors=(c, c, c), label_number=count_car)
                    needToRenderNumber = False
                    future_vehicles.append(future_xy)
                else:
                    if category['name'].split('.')[0] == 'human':
                        box.render(self.axes, view=self.view, colors=(c_human, c_human, c_human))
                    else:
                        box.render(self.axes, view=self.view, colors=(c_barrier, c_barrier, c_barrier))

                # print(future_xy)
                # self._render_traj(future_xy, line_color=c, dot_color=(0, 0, 0))
                # change_xy= [(0.48770607521093906, -9.825462948917355), (0.5, -7.8), (0.5, -5.8), (0.5, -3.8), (0.5, -1.8), (0.5, -0.5), (0.5, 0.0)]
                # change_xy=np.array(change_xy)
                # self._render_traj(change_xy, line_color=c, dot_color=(0, 0, 0))

                # 
                

        # 
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])

        #  JSON 
        self.save_future_positions_to_json(future_ego_xy_local, future_vehicles)
        

    def render_hd_map(self, nusc, nusc_maps, sample_token):
        sample_record = nusc.get('sample', sample_token)
        sd_rec = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                                sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        info = {
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'scene_token': sample_record['scene_token']
        }

        layer_names = ['road_divider', 'road_segment', 'lane_divider',
                       'lane',  'road_divider', 'traffic_light', 'ped_crossing']
        layer_names = [ 'lane_divider','road_divider']
        map_mask = obtain_map_info(nusc,
                                    nusc_maps,
                                    info,
                                    patch_size=(102.4, 102.4),
                                    canvas_size=(1024, 1024),
                                    layer_names=layer_names)
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = map_mask[:, ::-1] > 0
        map_show = np.ones((1024, 1024, 3))
        map_show[map_mask[0], :] = np.array([1.00, 0.50, 0.31])
        map_show[map_mask[1], :] = np.array([159./255., 0.0, 1.0])
        map_show[map_mask[2], :] = np.array([1, 1, 0])
        self.axes.imshow(map_show, alpha=0.3, interpolation='nearest',
                            extent=(-51.2, 51.2, -51.2, 51.2))
        # self.render_ego_vehicle(nusc, sample_token)

    def save_fig(self, filename):
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        print(f'saving to {filename}')
        plt.savefig(filename)
    def close_canvas(self):
        plt.close()
    def set_plot_cfg(self):
        """
        
        """
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        self.axes.set_aspect('equal')
        self.axes.grid(False)
    def reset_canvas(self, dx=1, dy=1, tight_layout=False):
        plt.close()
        plt.gca().set_axis_off()
        plt.axis('off')
        self.fig, self.axes = plt.subplots(dx, dy, figsize=self.figsize)
        if tight_layout:
            plt.tight_layout()
    def _render_traj(self, future_traj, traj_score=1, colormap='winter', points_per_step=20, line_color=None, dot_color=None, dot_size=25):
        total_steps = (len(future_traj)-1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors*traj_score + \
            (1-traj_score)*np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps-1):
            unit_vec = future_traj[i//points_per_step +
                                   1] - future_traj[i//points_per_step]
            total_xy[i] = (i/points_per_step - i//points_per_step) * \
                unit_vec + future_traj[i//points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)


# dataroot = '/data2//nuscenes/v1.0-mini'
dataroot = '/data2//nuscenes'

output_dataset=dataroot.split('/')[-1]
# nuscenes = NuScenes('v1.0-mini', dataroot=dataroot)
nuscenes = NuScenes('v1.0-trainval', dataroot=dataroot)

predict_helper = PredictHelper(nuscenes)
nusc_maps = {
                'boston-seaport': NuScenesMap(dataroot=dataroot, map_name='boston-seaport'),
                'singapore-hollandvillage': NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage'),
                'singapore-onenorth': NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth'),
                'singapore-queenstown': NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown'),
            }

scene_splits = create_splits_scenes()
# print(scene_splits)
train_split = scene_splits['train']
val_split = scene_splits['val']
train_list = []
val_list = []
for i,scene in enumerate(nuscenes.scene):
    print(i)
    # print(scene['name'])

    if scene['name'] in train_split:
        train_list.append(i)
    elif scene['name'] in val_split:
        val_list.append(i)
#sample

#/data2//nuscenes_trainval_1.0/nuscenes_infos_train.pkl pkl
# mmdetection_infos_path='/data2//nuscenes_trainval_1.0/nuscenes_infos_train.pkl'
# with open(mmdetection_infos_path, 'rb') as f:
#     mmdetection_infos = pickle.load(f)
# mmdetection_data_list=mmdetection_infos['data_list']
random_seed=3
random.seed(random_seed)

data=500
already_data_path="/home//VLM_attack_propose/annotation/mini-data_250_val_random3_bug_fix_before_20_frames.json"
data_save_path="/home//VLM_attack_propose/annotation/mini-data_500_continue_random3_bug_fix_before_20_frames.json"

collision_type_list=['A vehicle cuts in and collides with the ego vehicle','A vehicle rear-ends the ego vehicle','Ego vehicle rear-ends another vehicle','A vehicle has a head-on collision with the ego vehicle','A vehicle has a T-bone collision with the ego vehicle']
annotations=[]
with open(already_data_path, 'r', encoding='utf-8') as f:
    already_data = json.load(f)
while len(annotations)<data*5:
    #train
    meta_annotation={
    'sample_token':None,
    'bev_path':None,
    'rgb_path':None,
    'collision':'',
    'ego_init_speed':0,
    'reward':{},
    'token':{}
    }
    output_scene_number=random.choice(val_list)
    scene = nuscenes.scene[output_scene_number]
    #
    output_sample_number=random.choice(range(1,scene['nbr_samples']-20))
    # output_scene_number=394
    # scene = nuscenes.scene[output_scene_number]
    # output_sample_number=86

    sample_token = scene['first_sample_token']
    count=output_sample_number#3 10 
    for i in range(count):
        sample_token = nuscenes.get('sample', sample_token)['next']
    meta_annotation['sample_token']=sample_token
    already_in_annotation=False
    for annotation in annotations:
        if annotation['sample_token']==sample_token :
            already_in_annotation=True
            break
    meta_annotation['ego_init_speed']=get_ego_speed_from_sample(nuscenes, sample_token)
    already_in_annotation=False
    for annotation in annotations:
        if annotation['sample_token']==sample_token:
            already_in_annotation=True
            break
    for already in already_data:
        if already['sample_token']==sample_token:
            already_in_annotation=True
            break
    if already_in_annotation:
        continue
    # nuscenes
    #annotation
    annos_tokens = nuscenes.get('sample', sample_token)['anns']
    #
    lidar_sample_data = nuscenes.get('sample_data',nuscenes.get('sample', sample_token)['data']['LIDAR_TOP'])
    sd_ep = nuscenes.get("ego_pose", lidar_sample_data["ego_pose_token"])
    ego_translation = sd_ep['translation']

    anno_need_to_render_dict={}
    annos_tokens = sorted(annos_tokens, key=lambda anno_token: np.max(np.linalg.norm(np.array(ego_translation)-np.array(nuscenes.get('sample_annotation', anno_token)['translation']))))
    count=0
    for anno_token in annos_tokens:
        anno=nuscenes.get('sample_annotation', anno_token)
        if not (anno['category_name'].split('.')[0] == 'vehicle' and anno['category_name'].split('.')[1]!='bicycle' and anno['category_name'].split('.')[1]!='motorcycle'):
            continue
        distance=np.linalg.norm(np.array(ego_translation)-np.array(anno['translation']))
        visibility_token=anno['visibility_token']
        # visibility=nuscenes.get('visibility', visibility_token)
        # print(visibility)
        
        if distance>25:
            continue
        count=count+1
        if distance>5 and visibility_token!='4':
            continue
        anno_need_to_render_dict[""+str(count)]=(anno_token)
        
        print(distance)
    # mmdetectiontoken,todo
    # sample_idx = next((i['sample_idx'] for i in mmdetection_data_list if i['token'] == sample_token), 0)
    # instances = mmdetection_data_list[sample_idx]['instances']
    # for instance in instances:
    #     x, y= instance['bbox_3d'][:2]
    #     distance=np.linalg.norm(np.array([x,y]))
    #     if distance>30:
    #         continue


    bev_render=BEVRender()
    bev_render.reset_canvas(dx=1, dy=1)
    bev_render.set_plot_cfg()
    bev_render.render_anno_data(
                    sample_token, nuscenes, predict_helper)
    bev_render.render_sdc_car()
    bev_render.render_hd_map(
        nuscenes, nusc_maps, sample_token)
    rgb_image_dir='/home//VLM_attack_propose/rgb_image'
    if not os.path.exists(rgb_image_dir):
        os.makedirs(rgb_image_dir)
    view_list=['CAM_FRONT_LEFT','CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
    file_list=[]
    for i,view in enumerate(view_list):
        sample_data_token = nuscenes.get('sample', sample_token)['data'][view]
        sample_data = nuscenes.get('sample_data', sample_data_token)
        #rgb_image_dir
        shutil.copy(os.path.join(dataroot, sample_data['filename']), os.path.join(rgb_image_dir, view+".jpg"))
        file_list.append(os.path.join(rgb_image_dir, view+".jpg"))

    images={}

    for i,file in enumerate(file_list):
        image=Image.open(file)
        #2D box
        draw = ImageDraw.Draw(image)

        for label,anno in anno_need_to_render_dict.items():
            # _, boxes, _ = nuscenes.get_sample_data(nuscenes.get('sample', sample_token)['data'][view_list[i]], box_vis_level=BoxVisibility.ANY,
            #                                             selected_anntokens=[anno])
            # if len(boxes) == 0:
            #     continue
            x_box,y_box,height,width=get_2d_center_in_camera(anno,nuscenes,view_list[i])
            
            if x_box == None:
                continue
            #
            image_width, image_height = image.size
            x_center_pixel = x_box
            y_center_pixel = y_box
            box_width_pixel = width
            box_height_pixel = height

            # 
            x1 = x_center_pixel - box_width_pixel / 2
            y1 = y_center_pixel - box_height_pixel / 2
            x2 = x_center_pixel + box_width_pixel / 2
            y2 = y_center_pixel + box_height_pixel / 2

            # 
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        
            # 
            label_text = f"{label}"
            font = ImageFont.load_default(70)

            if y1 - 30 >= 0:
               
                #  textbbox  bounding box
                text_bbox = draw.textbbox((x1, y1 - 30), label_text, font)
                text_width = text_bbox[2] - text_bbox[0]  # 
                text_height = text_bbox[3] - text_bbox[1]  # 
                
                # 
                draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="red")
                
                # 
                draw.text((x1, y1-text_height-18), label_text, fill="white", font=font)
            else:
                #  textbbox  bounding box
                text_bbox = draw.textbbox((x1, y1 + 30), label_text, font)
                text_width = text_bbox[2] - text_bbox[0]  # 
                text_height = text_bbox[3] - text_bbox[1]  # 
                # 
                draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill="red")

                # 
                draw.text((x1, y1-18), label_text, fill="white", font=font)

            
        #resize896*448
        # if "BACK" in file:
        #     #180
        #     image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = image.resize((896, 448))
        images[i]=image
    for label,anno in anno_need_to_render_dict.items():
        meta_annotation['reward'][f"{label}"] = 0
        meta_annotation['token'][f"{label}"] = nuscenes.get('sample_annotation', anno)["instance_token"]
    
    border_colors = [
        (0, 255, 255),  # black
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0), # Yellow
        (255, 165, 0), # Orange
        (255, 105, 180) # Hot Pink
    ]
    width, height = images[0].size
    new_image = Image.new('RGB', (width * 3 + 12, height * 2 + 8), (255, 255, 255))  # 
    for i in range(2):
        for j in range(3):

            img = images[i * 3 + j]
            # 
            border_color = border_colors[i * 3 + j]
            bordered_img = Image.new('RGB', (width + 4, height + 4), border_color)  # 
            bordered_img.paste(img, (2, 2))  # 

            # 
            new_image.paste(bordered_img, (j * (width + 4), i * (height + 4)))

            # 
            draw = ImageDraw.Draw(new_image)
            font_size = 36
            font = ImageFont.load_default(font_size)
            
            text = f'{view_list[i * 3 + j]}'  # 
            draw.text((j * (width + 4) + 10, i * (height + 4) + 10), text, font=font, fill=(255, 255, 255))
    #save'/home//VLM_attack_propose/example_total/' + output_dataset + '_'+str(output_scene_number)+'_'+str(output_sample_number)+'.jpg'
    if not os.path.exists('/home//VLM_attack_propose/example_bev_500_continue_val_random3_bug_fix_before_20_frames/'):
        os.makedirs('/home//VLM_attack_propose/example_bev_500_continue_val_random3_bug_fix_before_20_frames/')
    
    if  not os.path.exists('/home//VLM_attack_propose/example_rgb_500_continue_val_random3_bug_fix_before_20_frames/'):
        os.makedirs('/home//VLM_attack_propose/example_rgb_500_continue_val_random3_bug_fix_before_20_frames/')

    new_image.save('/home//VLM_attack_propose/example_rgb_500_continue_val_random3_bug_fix_before_20_frames/' + output_dataset + '_'+str(output_scene_number)+'_'+str(output_sample_number)+'.jpg')
    # for i in range(6):
    #     rgb_image = nuscenes.get('sample_data', sample_token)['filename']
    #     rgb_image = os.path.join(dataroot, rgb_image)
    #     shutil.copy(rgb_image, os.path.join(rgb_image_dir, rgb_image.split('/')[-1]))
    bev_render.save_fig('/home//VLM_attack_propose/example_bev_500_continue_val_random3_bug_fix_after6frames' + output_dataset + '_'+str(output_scene_number)+'_'+str(output_sample_number)+'.jpg')

    meta_annotation['bev_path'] = '/home//VLM_attack_propose/example_bev_500_continue_val_random3_bug_fix_before_20_frames/' + output_dataset + '_'+str(output_scene_number)+'_'+str(output_sample_number)+'.jpg'
    meta_annotation['rgb_path'] = '/home//VLM_attack_propose/example_rgb_500_continue_val_random3_bug_fix_before_20_frames/' + output_dataset + '_'+str(output_scene_number)+'_'+str(output_sample_number)+'.jpg'

    for i in collision_type_list:
        
        new_meta_annotation = copy.deepcopy(meta_annotation)
        new_meta_annotation['collision'] =i
        annotations.append(new_meta_annotation)
#annotation/home//VLM_attack_propose/example_bev_val/nuscenes_trainval_1.0_11_24.jpg
annotations=sorted(annotations,key=lambda x:(int(x['bev_path'].split('_')[-2]),int(x['bev_path'].split('_')[-1].split('.')[0])))
with open(data_save_path, 'w') as f:
    json.dump(annotations, f, indent=4)
    