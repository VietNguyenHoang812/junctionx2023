#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
from shapely.geometry import Polygon
import cv2
import matplotlib.pyplot as plt

from os.path import join as join
from scipy.spatial import ConvexHull
from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


def camera_stream(path):
    #cams = os.listfir(path)
    list_cams = ['01', '02', '03', '04']
    fr_arr =os.listdir(join(path, '01'))
    num_frames = len(fr_arr)
    
    for fr in fr_arr:
        img1_path = join(path, '01', fr)
        img2_path = join(path, '02', fr)
        img3_path = join(path, '03', fr)
        img4_path = join(path, '04', fr)

        


    return


if __name__ == '__main__':
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

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with from scipy.spatial import ConvexHullt use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # Load the SuperPoint and SuperGlue models.
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

    # Create the output directories if they do not exist already.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)
    #print("----------------------------------------")
    #print(pairs)
    '''for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # Handle --cache logic.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        if opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                do_match = False
            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path)
                err_R, err_t = results['error_R'], results['error_t']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']
                do_eval = False
            if opt.viz and viz_path.exists():
                do_viz = False
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, opt.resize, rot1, opt.resize_float)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')

        #------------------------------------------------ load input------------------------------------------------------------------





        #------------------------------------------------------------------------------------------------------------------

        # inp0= np.expand_dims(inp0,0)
        # inp0=torch.from_numpy(np.concatenate((inp0,inp0), axis=0))
        # inp1= np.expand_dims(inp1,0)
        # inp1=torch.from_numpy(np.concatenate((inp1,inp1), axis=0))
        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            print("+++++++++++++++++++++++++here++++++++++++++++++++++++++++")
            #-------------------------------------------------------------
            #np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints, shape = (n_kpts, 2) 
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        # print("::::::::::::::::::::::::::::::::::")
        # print("m0: ", mkpts0)
        # print("m1: ", mkpts1)
        # print(mkpts0.shape, mkpts1.shape)
        if do_eval:
            # Estimate the pose and compute the pose error.
            assert len(pair) == 38, 'Pair does not have ground truth info'
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # Scale the intrinsics to resized image.
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # Update the intrinsics + extrinsics if EXIF rotation was found.
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0

            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            thresh = 1.  # In pixels relative to resized image size.
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            if ret is None:rot0, rot1 = 0, 0
            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            
            #---------------------------------------------------------------------------------
            #np.savez(str(eval_path), **out_eval)
            timer.update('eval')

        if do_viz:
            #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        if do_viz_eval:
            # Visualize the evaluation results for the image pair.
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            text = [
                'SuperGlue',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info (only works with --fast_viz).
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, text, viz_eval_path,
                opt.show_keypoints, opt.fast_viz,
                opt.opencv_display, 'Relative Pose', small_text)

            timer.update('viz_eval')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    if opt.eval:
        # Collate the results into a final table and print to terminal.
        pose_errors = []
        precisions = []
        matching_scores = []
        for pair in pairs:
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100.*yy for yy in aucs]
        prec = 100.*np.mean(precisions)
        ms = 100.*np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))'''


    #----------------------------------------------- self----------------------------------------------
    def matching_cam(x, y, img_x, img_y, stem_x, stem_y, viz_path, name1='0', name2='1'):
        pred = matching({'image0': x, 'image1': y})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]
        # if rot0 != 0 or rot1 != 0:
        #     text.append('Rotation: {}:{}'.format(rot0, rot1))

        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {}:{}'.format(stem_x, stem_y),
        ]

        make_matching_plot(
            img_x, img_y, kpts0, kpts1, mkpts0, mkpts1, color,
            text, viz_path, True,
            opt.fast_viz, opt.opencv_display, 'Matches', small_text)


        return mkpts0, mkpts1


    #-----------------------------------------------------
    def get_polygon(hull, mkpts):
        idx_points = hull.simplices
        list_pts =[]
        x = mkpts[hull.vertices, 0]
        y = mkpts[hull.vertices, 1]
        for i in range(len(x)):
            list_pts.append((x[i], y[i]))
        return Polygon(list_pts)

    #-----------------------------------------------------
           
    
    
    list_cams = os.listdir(input_dir)
    fr_arr =os.listdir(join(input_dir, '1'))
    num_frames = len(fr_arr)



    for fr in fr_arr:
        list_imgs = []
        for cam in list_cams:
            list_imgs
        # img1_path = join(input_dir, cam, fr)
        # img2_path = join(input_dir, cam, fr)
        # img3_path = join(input_dir, cam, fr)
        # img4_path = join(input_dir, '4', fr)

        rot0=0
        # print("<<<<<<<<<<<<<<<<<<")
        # print(img1_path)
        
        stem1, stem2, stem3, stem4 = Path('1_'+fr).stem, Path('2_'+fr).stem, \
            Path('3_'+fr).stem, Path('4_'+fr).stem
        viz_path_1 = output_dir / '{}_{}_matches.{}'.format(stem1, stem2, opt.viz_extension)
        viz_path_2 = output_dir / '{}_{}_matches.{}'.format(stem2, stem3, opt.viz_extension)
        viz_path_3 = output_dir / '{}_{}_matches.{}'.format(stem3, stem4, opt.viz_extension)
        viz_path_4 = output_dir / '{}_{}_matches.{}'.format(stem4, stem1, opt.viz_extension)



        img1, inp1, scales1= read_image(
            img1_path, device, opt.resize, rot0, opt.resize_float
        )
        img2, inp2, scales2= read_image(
            img2_path, device, opt.resize, rot0, opt.resize_float
        )
        img3, inp3, scales3= read_image(
            img3_path, device, opt.resize, rot0, opt.resize_float
        )
        img4, inp4, scales4= read_image(
            img4_path, device, opt.resize, rot0, opt.resize_float
        )
        
        # matching_cam(inp1, inp2, img1, img2, stem1, stem2, viz_path_1)
        # matching_cam(inp2, inp3, img2, img3, stem2, stem3, viz_path_2)
        # matching_cam(inp3, inp4, img3, img4, stem3, stem4, viz_path_3)
        # matching_cam(inp4, inp1, img4, img1, stem4, stem1, viz_path_4)

        mkpts_1_2, mkpts_2_1 = matching_cam(inp1, inp2, img1, img2, stem1, stem2, viz_path_1)
        mkpts_1_3, mkpts_3_1 = matching_cam(inp2, inp3, img2, img3, stem2, stem3, viz_path_2)
        mkpts_1_4, mkpts_4_1 = matching_cam(inp3, inp4, img3, img4, stem3, stem4, viz_path_3)
        #mkpts_1_2, mkpts_1_2 = matching_cam(inp4, inp1, img4, img1, stem4, stem1, viz_path_4)
        hull_1_2 = ConvexHull(mkpts_1_2)
        hull_1_3 = ConvexHull(mkpts_1_3)
        hull_1_4 = ConvexHull(mkpts_1_4)

        
        img1_ori = cv2.resize(cv2.imread(img1_path), (512,288))



        #print(">>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
        # print(mkpts_1_2[hull_1_2.vertices, 0])
        # print(mkpts_1_2[hull_1_2.vertices, 1])
        # print(hull_1_2.simplices)
        poly1= get_polygon(hull_1_2, mkpts_1_2)
        poly2 = get_polygon(hull_1_3, mkpts_1_3)
        poly3 = get_polygon(hull_1_4, mkpts_1_4)
        interst = poly1.intersection(poly2).intersection(poly3)
        x,y = interst.exterior.xy
        # print("+++++++++++++++++++++++++")
        # print(img1.shape)
        plt.imshow(img1_ori)
        plt.plot(x,y)
        plt.savefig('polygon_1.jpg')

        '''exit()
        
    

        import matplotlib.pyplot as plt
        fig, (ax1) = plt.subplots(ncols=1, figsize=(40, 12))
        ax1.imshow(img1)

        # ax1.set_title('Convex hull')
        for simplex in hull_1_2.simplices:
            ax1.plot(mkpts_1_2[simplex, 0], mkpts_1_2[simplex, 1], 'c')
        ax1.plot(mkpts_1_2[hull_1_2.vertices, 0], mkpts_1_2[hull_1_2.vertices, 1], 'o', mec='r', color='r', lw=1, markersize=10)
        ax1.set_xticks(range(10))
        ax1.set_yticks(range(10))

        for simplex in hull_1_3.simplices:
            ax1.plot(mkpts_1_3[simplex, 0], mkpts_1_3[simplex, 1], 'c')
        ax1.plot(mkpts_1_3[hull_1_3.vertices, 0], mkpts_1_3[hull_1_3.vertices, 1], 'o', mec='r', color='g', lw=1, markersize=10)
        ax1.set_xticks(range(10))
        ax1.set_yticks(range(10))

        for simplex in hull_1_4.simplices:
            ax1.plot(mkpts_1_4[simplex, 0], mkpts_1_4[simplex, 1], 'c')
        ax1.plot(mkpts_1_4[hull_1_4.vertices, 0], mkpts_1_4[hull_1_4.vertices, 1], 'o', mec='r', color='b', lw=1, markersize=10)
        ax1.set_xticks(range(10))
        ax1.set_yticks(range(10))
        # for ax in (ax1, ax2):
        #     # ax.plot(mkpts0[:, 0], mkpts0[:, 1], '.', color='k')
        #     # if ax == ax1:
        #     #     ax.set_title('Given points')
        #     # else:
        #     ax.set_title('Convex hull')
        #     for simplex in hull.simplices:
        #         ax.plot(mkpts0[simplex, 0], mkpts0[simplex, 1], 'c')
        #     ax.plot(mkpts0[hull.vertices, 0], mkpts0[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
        #     ax.set_xticks(range(10))
        #     ax.set_yticks(range(10))
        # plt.show()

        plt.savefig('tmp.jpg')
        
        exit()'''
        


        #---------------------------------draw polygons------------------------------------------









        # print(pred1, pred2, pred3, pred4)
        # exit()
