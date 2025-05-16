
import os
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

def create_multi_view_video(nusc, scene_name, output_path, fps=12):
    """
    sample_tokenA
    """
    print(f": {scene_name}")
    
    # 
    scene = next((s for s in nusc.scene if s['name'] == scene_name), None)
    if scene is None:
        raise ValueError(f" {scene_name}")
    
    # 
    first_row = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
    second_row = ['CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
    all_cameras = first_row + second_row

    # 
    first_sample = nusc.get('sample', scene['first_sample_token'])
    first_cam_data = nusc.get('sample_data', first_sample['data']['CAM_FRONT'])
    img = cv2.imread(os.path.join(nusc.dataroot, first_cam_data['filename']))
    scale_factor = 0.5
    img_height, img_width = int(img.shape[0] * scale_factor), int(img.shape[1] * scale_factor)
    video_height, video_width = img_height * 2, img_width * 3

    # 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

    #  sample_data token
    cam_tokens = {
        cam: nusc.get('sample', scene['first_sample_token'])['data'][cam]
        for cam in all_cameras
    }

    # CAM_FRONT
    front_token = cam_tokens['CAM_FRONT']
    frames_processed = 0
    progress_bar = tqdm(desc=f" {scene_name}", unit="")

    while front_token:
        ref_data = nusc.get('sample_data', front_token)
        timestamp = ref_data['timestamp']
        
        # 
        frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)

        # 
        for i, cam in enumerate(all_cameras):
            row = 0 if cam in first_row else 1
            col = first_row.index(cam) if row == 0 else second_row.index(cam)

            try:
                # 
                cam_token = cam_tokens[cam]
                cam_data = nusc.get('sample_data', cam_token)

                # 
                while cam_data['timestamp'] < timestamp and cam_data['next']:
                    cam_data = nusc.get('sample_data', cam_data['next'])

                # token
                cam_tokens[cam] = cam_data['next'] if cam_data['next'] else cam_data['token']

                img_path = os.path.join(nusc.dataroot, cam_data['filename'])
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f": {img_path}")

                img = cv2.resize(img, (img_width, img_height))
                y_start = row * img_height
                x_start = col * img_width
                frame[y_start:y_start+img_height, x_start:x_start+img_width] = img

                # 
                cv2.putText(frame, cam, (x_start + 10, y_start + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"{cam} : {e}")
                error_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                cv2.putText(error_img, f"Error: {cam}", (10, img_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y_start = row * img_height
                x_start = col * img_width
                frame[y_start:y_start+img_height, x_start:x_start+img_width] = error_img

        # 
        time_sec = timestamp / 1e6
        info_text = f"Frame: {frames_processed} | Time: {time_sec:.3f}s"
        cv2.putText(frame, info_text, (10, video_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        video.write(frame)
        frames_processed += 1
        progress_bar.update(1)

        # 
        if not ref_data['next']:
            break
        front_token = ref_data['next']

    video.release()
    print(f": {output_path}")

def main():
    # 
    dataroot = '/data3//OpenDWM/OpenDWM_test_dataset/5_2_adversarial_trajectory_250_val_41_items_first_stage'  # NuScenes
    version = 'v1.0-collision'  #  'v1.0-trainval'
    output_dir = '/data3//OpenDWM/OpenDWM_test_dataset_convert_video/5_2_adversarial_trajectory_250_val_41_items_first_stage'  # 
    
    # 
    os.makedirs(output_dir, exist_ok=True)
    
    # NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    
    # 
    print(":")
    for i, scene in enumerate(nusc.scene):
        print(f"{i+1}. {scene['name']}")
    
    # 
    process_choice = input("(s)(a)? ")
    
    if process_choice.lower() == 's':
        # 
        scene_index = int(input(": ")) - 1
        scene_name = nusc.scene[scene_index]['name']
        scene_token = nusc.scene[scene_index]['token']
        output_path = os.path.join(output_dir, f"{scene_name}.mp4")
        create_multi_view_video(nusc, scene_name, output_path)
    else:
        # 
        print("...")
        for i, scene in enumerate(tqdm(nusc.scene)):
            scene_name = scene['name']
            output_path = os.path.join(output_dir, f"{scene_name}.mp4")
            try:
                create_multi_view_video(nusc, scene_name, output_path)
                print(f": {scene_name}")
            except Exception as e:
                print(f" {scene_name} : {e}")

def process_specific_scene(nusc, scene_token, output_dir):
    """
    
    
    :
        nusc: NuScenes
        scene_token: token
        output_dir: 
    """
    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']
    output_path = os.path.join(output_dir, f"{scene_name}.mp4")
    create_multi_view_video(nusc, scene_name, output_path, fps=12)

if __name__ == "__main__":
    main()