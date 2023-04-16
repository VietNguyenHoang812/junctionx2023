from xxlimited import Str
import cv2
import os
from typing import List
import argparse
from SuperGluePretrainedNetwork.match_pairs import run_app
from SuperGluePretrainedNetwork.models.matching import Matching
import torch
import numpy as np


def make_dir(dir_path: str) -> None:
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    else:
        pass

def get_list_videos(list_vid_pathfiles: List[str]) -> list:
    list_videos = []
    for vid_pathfile in list_vid_pathfiles:
        video = cv2.VideoCapture(vid_pathfile)
        list_videos.append(video)

    return list_videos

def get_multi_frames(list_videos: list) -> list:
    list_frames = []
    for video in list_videos:
        frame = get_present_frame(video)
        list_frames.append(frame)

    return list_frames

def get_present_frame(vid):
    success, frame = vid.read()
    if not success:
        return None
    return frame



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--alert_outline', action='store_true', default=False)
    parser.add_argument(
        '--count_outline', action='store_true', default=False)



    opt = parser.parse_args()
    print(opt)
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    vid1_pathfile = "/home/quyet0nguyen/Documents/hackathon/hackathon_overlapping_area_estimation/dataset/original/1.mp4"
    vid2_pathfile = "/home/quyet0nguyen/Documents/hackathon/hackathon_overlapping_area_estimation/dataset/original/2.mp4"
    vid3_pathfile = "/home/quyet0nguyen/Documents/hackathon/hackathon_overlapping_area_estimation/dataset/original/3.mp4"
    vid4_pathfile = "/home/quyet0nguyen/Documents/hackathon/hackathon_overlapping_area_estimation/dataset/original/4.mp4"
    # vid1_pathfile = "rtsp://192.168.5.101/main"
    # vid2_pathfile = "rtsp://192.168.5.102/main"
    # vid3_pathfile = "rtsp://192.168.5.103/main"
    # vid4_pathfile = "rtsp://192.168.5.104/main"
    list_vid_pathfiles = [vid1_pathfile, vid2_pathfile, vid3_pathfile, vid4_pathfile]

    make_dir("output")
    list_video_frames_dir = []
    for i in range(len(list_vid_pathfiles)):
        video_frames_dir = f"output/{i+1}"
        list_video_frames_dir.append(video_frames_dir)
        make_dir(video_frames_dir)

    count = 0
    count_false = 0
    thresh_count_false = 10
    list_videos = get_list_videos(list_vid_pathfiles)
    num_to_save_res = 0
    list_inters = []
    while(count_false <= thresh_count_false):
        num_to_save_res += 1
        # os.makedirs('SuperGluePretrainedNetwork/output_demo/'+str(num_to_save_res), exist_ok=True)
        print(num_to_save_res)
        list_frames = get_multi_frames(list_videos)

        # Only save image if all get all images from each camera succesfully
        if any(frame is None for frame in list_frames):
            count_false += 1
            print("Count false: ", count_false)
            continue

        frame1, frame2, frame3, frame4 = list_frames
        if num_to_save_res%50 == 10:
            list_inters = run_app(opt, device, matching, frame1, frame2, frame3, frame4, num_to_save_res)
        if list_inters != []:
            for i,inter in enumerate(list_inters):
                if inter is not None:
                    if not inter.is_empty:
                        isClosed = True
                        list_frames[i] = cv2.resize(list_frames[i], (512,288))
                        print(inter)
                        xx, yy = inter.exterior.coords.xy
                        inter_array = []
                        for j in range(len(xx)):
                            inter_array.append([xx[j], yy[j]])
                        inter_array = np.array(inter_array, np.int32)
                        inter_array = inter_array.reshape((-1, 1, 2))
                        # Green color in BGR
                        print(inter)
                        color = (0, 255, 0)
                        
                        # Line thickness of 8 px
                        thickness = 8
                        
                        # Using cv2.polylines() method
                        # Draw a Green polygon with
                        # thickness of 1 px
                        image = cv2.polylines(list_frames[i], [inter_array],
                                            isClosed, color,
                                            thickness)
                        cv2.imwrite('SuperGluePretrainedNetwork/output_demo/'+ str(i) +'.jpg', image)
        # Show images
        # cv2.imshow("frame", frame1)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
            # break

        # cv2.imwrite(f"output/1/frame{count}.jpg", frame1)
        # cv2.imwrite(f"output/2/frame{count}.jpg", frame2)
        # cv2.imwrite(f"output/3/frame{count}.jpg", frame3)
        # cv2.imwrite(f"output/4/frame{count}.jpg", frame4)
        
        # # Reset count false
        # count_false = 0
        # count += 1

        #-------------------------------
        
    