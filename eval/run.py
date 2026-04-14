import os
import sys
import numpy as np
import torch
from tqdm import trange
import json
import glob
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2
import pandas as pd

from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.tools import plot

import matplotlib.pyplot as plt
import copy

calc_psnr = PeakSignalNoiseRatio().cuda()
calc_ssim = StructuralSimilarityIndexMeasure().cuda()
calc_lpips = LearnedPerceptualImagePatchSimilarity().cuda()


def create_error_visualization(render_image, gt_image, psnr_value, lpips_value, save_path):
    """Create and save error visualization showing where PSNR differences come from."""
    # Convert to numpy if needed
    if torch.is_tensor(render_image):
        render_np = render_image.detach().cpu().numpy()
    else:
        render_np = render_image
    
    if torch.is_tensor(gt_image):
        gt_np = gt_image.detach().cpu().numpy()
    else:
        gt_np = gt_image
    
    # Ensure images are in [0, 1] range
    render_np = np.clip(render_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    
    # Create error heatmap - simplified to 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original rendered image
    axes[0].imshow(render_np)
    axes[0].set_title('Rendered Image')
    axes[0].axis('off')
    
    # Ground truth image
    axes[1].imshow(gt_np)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Absolute difference heatmap
    abs_diff = np.abs(render_np - gt_np)
    abs_diff_gray = np.mean(abs_diff, axis=2)
    im = axes[2].imshow(abs_diff_gray, cmap='hot', vmin=0, vmax=0.2)  # Fixed scale for consistency
    axes[2].set_title(f'Error Map\nPSNR: {psnr_value:.2f} dB | LPIPS: {lpips_value:.3f}')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Reduced DPI for speed
    plt.close(fig)


def loadReplica(path):
    color_paths = sorted(glob.glob(os.path.join(path, "results/frame*.jpg")))
    tstamp = [
        float(
            color_path.split("/")[-1]
            .replace("frame", "")
            .replace(".jpg", "")
            .replace(".png", "")
        )
        for color_path in color_paths
    ]
    return color_paths, tstamp


def loadTUM(path):
    if os.path.exists(os.path.join(path, "rgb3")):
        color_paths = sorted(glob.glob(os.path.join(path, "rgb3/*.png")))
    else:
        color_paths = sorted(glob.glob(os.path.join(path, "rgb/*.png")))
    tstamp = [
        float(
            color_path.split("/")[-1]
            .replace("frame", "")
            .replace(".jpg", "")
            .replace(".png", "")
        )
        for color_path in color_paths
    ]
    return color_paths, tstamp


def loadKITTI(path):
    color_paths = sorted(glob.glob(os.path.join(path, "image_2/*.png")))
    tstamp = np.loadtxt(
        os.path.join(path, "times.txt"), delimiter=" ", dtype=np.str_
    ).astype(np.float32)
    return color_paths, tstamp


def loadEuRoC(path):
    color_paths = sorted(glob.glob(os.path.join(path, "mav0/cam0/data/*.png")))
    tstamp = [np.float64(x.split("/")[-1][:-4]) / 1e9 for x in color_paths]
    return color_paths, tstamp


def associate_frames(tstamp_image, tstamp_pose, max_dt=0.1):
    """Pair images, depths, and poses."""
    associations = []
    
    for i, t in enumerate(tstamp_image):
        j = np.argmin(np.abs(tstamp_pose - t))
        if np.abs(tstamp_pose[j] - t) < max_dt:
            associations.append((i, j))
            
    print(f"Associated {len(associations)} frames out of {len(tstamp_image)} ground truth frames")
    return associations

def load_slowdown_factor(result_path, default=1.0):
    """Load slowdown factor from result path if available, otherwise return default value."""
    slowdown_path = os.path.join(result_path, "slowdown_factor.txt")
    if os.path.exists(slowdown_path):
        try:
            with open(slowdown_path, "r") as f:
                slowdown_factor = float(f.readline().strip())
            print(f"Loaded slowdown factor: {slowdown_factor}x from {slowdown_path}")
            return slowdown_factor
        except (ValueError, IOError) as e:
            print(f"Error loading slowdown factor: {e}. Using default: {default}x")
            return default
    else:
        print(f"No slowdown factor file found at {slowdown_path}. Using default: {default}x")
        return default


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="evaluation script parameters")
    parser.add_argument("result_path", type=str, default=None)
    parser.add_argument("gt_path", type=str, default=None)
    parser.add_argument("--correct_scale", action="store_true")
    parser.add_argument("--show_plot", action="store_true")
    parser.add_argument("--skip_trajectory_eval", action="store_true", 
                        help="Skip trajectory evaluation and use ground truth poses for rendering")
    parser.add_argument("--skip_error_vis", action="store_true",
                        help="Skip creating error visualization images (saves time)")
    args = parser.parse_args()
    dirs = os.listdir(args.result_path)
    # Find shutdown directory
    shutdown_name = None
    for file_name in dirs:
        if ("shutdown" in file_name):
            shutdown_name = file_name
            break
    if shutdown_name is None:
        sys.exit("No shutdown dir found, exiting...")

    shutdown_path = os.path.join(args.result_path, shutdown_name)
    print(f"Found shutdown directory: {shutdown_path}")

    # Load render time statistics
    render_time = np.loadtxt(
        os.path.join(shutdown_path, "render_time.txt"),
        delimiter=" ",
        dtype=np.str_,
    )
    render_time = render_time[:, 1].astype(np.float32)

    # Load number of gaussians
    num_gaussians = np.loadtxt(
        os.path.join(shutdown_path, "gaussianCount.txt"),
        delimiter=" ",
        dtype=np.str_,
    )
    num_gaussians = int(num_gaussians.item())

    # load gt (only need for trajectory evaluation)
    if "replica" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadReplica(args.gt_path)
    elif "kitti" in args.gt_path.lower():
        gt_color_paths, gt_tstamp = loadKITTI(args.gt_path)
    elif "euroc" in args.gt_path.lower():
        print(args.gt_path)
        gt_color_paths, gt_tstamp = loadEuRoC(args.gt_path)
    else:
        gt_color_paths, gt_tstamp = loadTUM(args.gt_path)

    # Load ground truth poses for either trajectory evaluation or direct rendering
    if "kitti" in args.gt_path.lower():
        def loadKITTIPose(gt_path):
            scene = gt_path.split("/")[-1]
            gt_file = gt_path.replace(scene, "poses/{}.txt".format(scene))
            pose_quat = []
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i].split()
                    # print(line)
                    c2w = np.array(list(map(float, line))).reshape(3, 4)
                    # print(c2w)
                    quat = np.zeros(7)
                    quat[:3] = c2w[:3, 3]
                    quat[3:] = Rotation.from_matrix(c2w[:3, :3]).as_quat()
                    pose_quat.append(quat)
            pose_quat = np.array(pose_quat)
            return pose_quat

        pose_quat = loadKITTIPose(args.gt_path)
        traj_ref = PoseTrajectory3D(
            positions_xyz=pose_quat[:, :3],
            orientations_quat_wxyz=pose_quat[:, 3:],
            timestamps=np.array(gt_tstamp),
        )
    elif "replica" in args.gt_path.lower():
        gt_file = os.path.join(args.gt_path, "pose_TUM.txt")
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    elif "euroc" in args.gt_path.lower():
        gt_file = os.path.join(
            args.gt_path, "mav0/state_groundtruth_estimate0/data.csv"
        )
        traj_ref = file_interface.read_euroc_csv_trajectory(gt_file)
        T_i_c0 = np.array(
            [
                [
                    0.0148655429818,
                    -0.999880929698,
                    0.00414029679422,
                    -0.0216401454975,
                ],
                [
                    0.999557249008,
                    0.0149672133247,
                    0.025715529948,
                    -0.064676986768,
                ],
                [
                    -0.0257744366974,
                    0.00375618835797,
                    0.999660727178,
                    0.00981073058949,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        traj_ref.transform(T_i_c0, True)
    else:
        gt_file = os.path.join(args.gt_path, "groundtruth.txt")
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)
        
    # Load slowdown factor (if exists)
    slowdown_factor = load_slowdown_factor(args.result_path)
    
    training_time_path = os.path.join(args.result_path, "training_time.txt")
    training_time = 0.0
    with open(training_time_path, "r") as f:
        training_time = float(f.readline().strip())

    # If not skipping trajectory eval, load estimated poses and evaluate them
    if not args.skip_trajectory_eval:
        pose_path = os.path.join(args.result_path, "CameraTrajectory_TUM.txt")
        traj_est = file_interface.read_tum_trajectory_file(pose_path)
        
        # If there's a slowdown factor, adjust timestamps before association
        if slowdown_factor != 1.0:
            # Make a deep copy to avoid modifying the original
            traj_est_adjusted = copy.deepcopy(traj_est)
            # Adjust timestamps: current_time / slowdown
            traj_est_adjusted.timestamps = traj_est_adjusted.timestamps / slowdown_factor
            # Use the adjusted trajectory for association
            traj_ref_sync, traj_est = sync.associate_trajectories(
                traj_ref, traj_est_adjusted, max_diff=0.1
            )
        else:
            # Use original trajectory
            traj_ref_sync, traj_est = sync.associate_trajectories(
                traj_ref, traj_est, max_diff=0.1
            )
        
        traj_ref_sync.align(traj_est, True)
        poses = traj_est.poses_se3
        tstamp = traj_est.timestamps
        
        result = main_ape.ape(
            traj_ref_sync,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=args.correct_scale,
        )
        result_rotation_part = main_ape.ape(
            traj_ref_sync,
            traj_est,
            est_name="rot",
            pose_relation=PoseRelation.rotation_part,
            align=True,
            correct_scale=args.correct_scale,
        )

        out_path = os.path.join(args.result_path, "metrics_traj.txt")
        with open(out_path, "w") as fp:
            fp.write(result.pretty_str())
            fp.write(result_rotation_part.pretty_str())
        print(result)

        if args.show_plot:
            traj_est_aligned = copy.deepcopy(traj_est)
            traj_est_aligned.align(traj_ref_sync, correct_scale=True)
            fig = plt.figure()
            traj_by_label = {
                "estimate (not aligned)": traj_est,
                "estimate (aligned)": traj_est_aligned,
                "reference": traj_ref_sync,
            }
            plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
            plt.show()
    else:
        # When skipping trajectory eval, use ground truth poses directly
        print("Skipping trajectory evaluation, using ground truth poses for rendering")
        poses = traj_ref.poses_se3
        tstamp = traj_ref.timestamps
        
        # Write a note to metrics file
        out_path = os.path.join(args.result_path, "metrics_traj.txt")
        with open(out_path, "w") as fp:
            fp.write("Trajectory evaluation skipped - using ground truth poses for rendering\n")


    image_dir = os.path.join(shutdown_path, "image")
    image_gt_dir = os.path.join(shutdown_path, "image_gt")

    # Get all rendered images (format: iteration_kfid_suffix.jpg)
    rendered_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    if len(rendered_files) == 0:
        sys.exit(f"No rendered images found in {image_dir}")

    # Create output directories
    os.makedirs(os.path.join(args.result_path, "image"), exist_ok=True)
    if not args.skip_error_vis:
        os.makedirs(os.path.join(args.result_path, "error_analysis"), exist_ok=True)

    # Lists to store results
    psnr_list, ssim_list, lpips_list = [], [], []
    detailed_results = []

    for rendered_path in trange(len(rendered_files),
                                desc=f"Computing metrics"):
        rendered_path = rendered_files[rendered_path]
        filename = os.path.basename(rendered_path)

        # Parse filename: iteration_kfid[_suffix].jpg
        name_no_ext = os.path.splitext(filename)[0]
        parts = name_no_ext.split('_')

        if len(parts) < 2:
            print(f"Warning: Could not parse filename {filename}")
            continue

        try:
            iteration = int(parts[0])
            kfid = int(parts[1])
            suffix = '_'.join(parts[2:]) if len(parts) > 2 else ''
        except ValueError:
            print(f"Warning: Could not parse filename {filename}")
            continue

        # Construct ground truth filename
        gt_filename = f"{iteration}_{kfid}{('_' + suffix) if suffix else ''}_gt.jpg"
        gt_path = os.path.join(image_gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found: {gt_path}")
            continue

        # Load rendered image
        rendered_img = cv2.imread(rendered_path)
        rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
        rendered_img = rendered_img.astype(np.float32) / 255.0
        rendered_torch = torch.from_numpy(rendered_img).float().permute(2, 0, 1).unsqueeze(0).cuda()

        # Load ground truth image
        gt_img = cv2.imread(gt_path)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = gt_img.astype(np.float32) / 255.0
        gt_torch = torch.from_numpy(gt_img).float().permute(2, 0, 1).unsqueeze(0).cuda()

        # Compute metrics
        val_psnr = calc_psnr(rendered_torch, gt_torch).item()
        val_ssim = calc_ssim(rendered_torch, gt_torch).item()
        val_lpips = calc_lpips(rendered_torch, gt_torch).item()

        psnr_list.append(val_psnr)
        ssim_list.append(val_ssim)
        lpips_list.append(val_lpips)

        # Store detailed results
        detailed_results.append({
            'filename': filename,
            'iteration': iteration,
            'kfid': kfid,
            'suffix': suffix,
            'psnr': val_psnr,
            'ssim': val_ssim,
            'lpips': val_lpips
        })

        # Create error visualization
        if not args.skip_error_vis:
            error_vis_path = os.path.join(
                args.result_path,
                "error_analysis",
                f"error_{filename.replace('.jpg', '.png')}"
            )
            create_error_visualization(
                rendered_img,
                gt_img,
                val_psnr,
                val_lpips,
                error_vis_path
            )

    # Convert results to arrays
    psnr_list = np.array(psnr_list)
    ssim_list = np.array(ssim_list)
    lpips_list = np.array(lpips_list)

    # Save individual metric arrays
    np.savetxt(os.path.join(shutdown_path, "psnr_recomputed.txt"), psnr_list)
    np.savetxt(os.path.join(shutdown_path, "ssim_recomputed.txt"), ssim_list)
    np.savetxt(os.path.join(shutdown_path, "lpips_recomputed.txt"), lpips_list)

    # Save detailed results as CSV
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(os.path.join(shutdown_path, "detailed_metrics.csv"), index=False)

    # Save summary statistics
    summary_stats = {
        'metric': ['psnr', 'ssim', 'lpips'],
        'mean': [np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)],
        'std': [np.std(psnr_list), np.std(ssim_list), np.std(lpips_list)],
        'min': [np.min(psnr_list), np.min(ssim_list), np.min(lpips_list)],
        'max': [np.max(psnr_list), np.max(ssim_list), np.max(lpips_list)],
        'median': [np.median(psnr_list), np.median(ssim_list), np.median(lpips_list)]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(shutdown_path, "metrics_summary.csv"), index=False)

    print(f"\nMetrics Summary:")
    print(f"PSNR:  {np.mean(psnr_list):.3f} ± {np.std(psnr_list):.3f} dB (range: {np.min(psnr_list):.3f} - {np.max(psnr_list):.3f})")
    print(f"SSIM:  {np.mean(ssim_list):.3f} ± {np.std(ssim_list):.3f} (range: {np.min(ssim_list):.3f} - {np.max(ssim_list):.3f})")
    print(f"LPIPS: {np.mean(lpips_list):.3f} ± {np.std(lpips_list):.3f} (range: {np.min(lpips_list):.3f} - {np.max(lpips_list):.3f})")

    # For compatibility with eval.py, save results to result_path as well
    if shutdown_path is not None:
        np.savetxt(os.path.join(args.result_path, "psnr.txt"), psnr_list)
        np.savetxt(os.path.join(args.result_path, "ssim.txt"), ssim_list)
        np.savetxt(os.path.join(args.result_path, "lpips.txt"), lpips_list)

        # Handle tracking time evaluation if file exists
        tracking_time_path = os.path.join(args.result_path, "TrackingTime.txt")
        if os.path.exists(tracking_time_path):
            with open(tracking_time_path, "r") as fin:
                tracking_time = fin.readlines()
            if len(tracking_time) > 3:
                tracking_time = np.array(tracking_time[:-3]).astype(np.float32)
                tracking_fps = 1 / np.mean(tracking_time) if np.mean(tracking_time) > 0 else 0
            else:
                tracking_time = np.array([0])
                tracking_fps = 0
        else:
            tracking_time = np.array([0])
            tracking_fps = 0

        with open(os.path.join(args.result_path, "eval.txt"), "w") as fout:
            fout.write("psnr: {}\n".format(np.mean(psnr_list)))
            fout.write("ssim: {}\n".format(np.mean(ssim_list)))
            fout.write("lpips: {}\n".format(np.mean(lpips_list)))
            fout.write("time s: {}\n".format(training_time))
            fout.write("rendering ms: {}\n".format(np.mean(render_time)))
            fout.write("rendering FPS: {}\n".format(1000 / np.mean(render_time)))
            fout.write("num gaussians: {}\n".format(num_gaussians))

        print(f"\nFiles saved to {args.result_path}:")
        print(f"- eval.txt (for eval.py compatibility)")
        print(f"- Individual metrics: psnr.txt, ssim.txt, lpips.txt")
        if not args.skip_error_vis:
            print(f"- Error visualizations: error_analysis/ folder")

    print("\nDone!")