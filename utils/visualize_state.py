import os
import torch
import numpy as np
import pypose as pp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _unwrap_euler_deg(rot):
    """Convert SO3 rotations to unwrapped Euler angles in degrees."""
    euler_rad = pp.SO3(rot).euler().detach().cpu().numpy()
    return np.rad2deg(np.unwrap(euler_rad, axis=0))


def visualize_state_error(save_prefix, relative_outstate, relative_infstate, \
                            save_folder=None, mask=None, file_name="state_error_compare.png"):
    if mask is None:
        outstate_pos_err = relative_outstate['pos_dist'][0]
        outstate_vel_err = relative_outstate['vel_dist'][0]
        outstate_rot_err = relative_outstate['rot_dist'][0]
        
        infstate_pos_err = relative_infstate['pos_dist'][0]
        infstate_vel_err = relative_infstate['vel_dist'][0]
        infstate_rot_err = relative_infstate['rot_dist'][0]
    else:
        outstate_pos_err = relative_outstate['pos_dist'][0, mask]
        outstate_vel_err = relative_outstate['vel_dist'][0, mask]
        outstate_rot_err = relative_outstate['rot_dist'][0, mask]
        
        infstate_pos_err = relative_infstate['pos_dist'][0, mask]
        infstate_vel_err = relative_infstate['vel_dist'][0, mask]
        infstate_rot_err = relative_infstate['rot_dist'][0, mask]
    
    fig, axs = plt.subplots(3,)
    fig.suptitle("Integration error vs AirIMU Integration error")
    
    axs[0].plot(outstate_pos_err,color = 'b',linewidth=1)
    axs[0].plot(infstate_pos_err,color = 'red',linewidth=1)
    axs[0].legend(["integration_pos_error", "AirIMU_pos_error"])
    axs[0].grid(True)
    
    axs[1].plot(outstate_vel_err,color = 'b',linewidth=1)
    axs[1].plot(infstate_vel_err,color = 'red',linewidth=1)
    axs[1].legend(["integration_vel_error", "AirIMU_vel_error"])
    axs[1].grid(True)
    
    axs[2].plot(outstate_rot_err,color = 'b',linewidth=1)
    axs[2].plot(infstate_rot_err,color = 'red',linewidth=1)
    axs[2].legend(["integration_rot_error", "AirIMU_rot_error"])
    axs[2].grid(True)
    
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, save_prefix + file_name), dpi = 300)
    plt.show()
  

def visualize_rotations(save_prefix, gt_rot, out_rot, inf_rot = None,save_folder=None):
   
    gt_euler = _unwrap_euler_deg(gt_rot)
    outstate_euler = _unwrap_euler_deg(out_rot)
    
    legend_list = ["roll","pitch", "yaw"]
    fig, axs = plt.subplots(3,)
    fig.suptitle("integrated orientation")
    for i in range(3):
        axs[i].plot(outstate_euler[:,i],color = 'b',linewidth=0.9)
        axs[i].plot(gt_euler[:,i],color = 'mediumseagreen',linewidth=0.9)
        axs[i].legend(["Integrated_"+legend_list[i],"gt_"+legend_list[i]])
        axs[i].grid(True)
    
    if inf_rot is not None:
        infstate_euler = _unwrap_euler_deg(inf_rot)
        print(infstate_euler.shape)
        for i in range(3):
            axs[i].plot(infstate_euler[:,i],color = 'red',linewidth=0.9)
            axs[i].legend(["Integrated_"+legend_list[i],"gt_"+legend_list[i],"AirIMU_"+legend_list[i]])
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, save_prefix+ "_orientation_compare.png"), dpi = 300)
    plt.show()


def _to_numpy_1d(tensor):
    return tensor.detach().cpu().numpy().reshape(-1)


def _compute_axis_limits(*arrays, margin_ratio=0.05):
    values = np.concatenate([np.asarray(arr).reshape(-1) for arr in arrays])
    min_v = np.min(values)
    max_v = np.max(values)
    span = max(max_v - min_v, 1e-6)
    margin = span * margin_ratio
    return min_v - margin, max_v + margin


def _plot_trajectory_projection(save_path, raw_x, raw_y, air_x, air_y, gt_x, gt_y, xlabel, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Full-scale trajectory view.
    axes[0].plot(raw_x, raw_y, label="Raw", linewidth=1.2)
    axes[0].plot(air_x, air_y, label="AirIMU", linewidth=1.2)
    axes[0].plot(gt_x, gt_y, label="Ground Truth", linewidth=1.2)
    axes[0].set_title("Full scale")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True)
    axes[0].set_xlim(*_compute_axis_limits(raw_x, air_x, gt_x))
    axes[0].set_ylim(*_compute_axis_limits(raw_y, air_y, gt_y))
    axes[0].legend()

    # Zoomed view focused on AirIMU and GT overlap.
    axes[1].plot(raw_x, raw_y, label="Raw", linewidth=1.0, alpha=0.25)
    axes[1].plot(air_x, air_y, label="AirIMU", linewidth=1.4)
    axes[1].plot(gt_x, gt_y, label="Ground Truth", linewidth=1.4)
    axes[1].set_title("Zoomed (AirIMU + GT region)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].grid(True)
    axes[1].set_xlim(*_compute_axis_limits(air_x, gt_x))
    axes[1].set_ylim(*_compute_axis_limits(air_y, gt_y))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_trajectory(save_prefix, save_folder, outstate, infstate):
    gt_x, gt_y, gt_z = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    raw_x, raw_y, raw_z = torch.split(outstate["poses"][0].cpu(), 1, dim=1)
    air_x, air_y, air_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)

    gt_x, gt_y, gt_z = _to_numpy_1d(gt_x), _to_numpy_1d(gt_y), _to_numpy_1d(gt_z)
    raw_x, raw_y, raw_z = _to_numpy_1d(raw_x), _to_numpy_1d(raw_y), _to_numpy_1d(raw_z)
    air_x, air_y, air_z = _to_numpy_1d(air_x), _to_numpy_1d(air_y), _to_numpy_1d(air_z)

    _plot_trajectory_projection(
        os.path.join(save_folder, save_prefix + "_trajectory_xy.png"),
        raw_x, raw_y, air_x, air_y, gt_x, gt_y,
        "X axis", "Y axis"
    )

    _plot_trajectory_projection(
        os.path.join(save_folder, save_prefix + "_trajectory_xz.png"),
        raw_x, raw_z, air_x, air_z, gt_x, gt_z,
        "X axis", "Z axis"
    )

    _plot_trajectory_projection(
        os.path.join(save_folder, save_prefix + "_trajectory_yz.png"),
        raw_y, raw_z, air_y, air_z, gt_y, gt_z,
        "Y axis", "Z axis"
    )

    fig = plt.figure(figsize=(13, 5), dpi=300)
    ax_full = fig.add_subplot(121, projection='3d')
    ax_zoom = fig.add_subplot(122, projection='3d')

    for ax in (ax_full, ax_zoom):
        ax.view_init(20, 30)
        ax.plot(raw_x, raw_y, raw_z, label="Raw", linewidth=1.0, alpha=0.25)
        ax.plot(air_x, air_y, air_z, label="AirIMU", linewidth=1.4)
        ax.plot(gt_x, gt_y, gt_z, label="Ground Truth", linewidth=1.4)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

    ax_full.set_title('3D full scale')
    ax_full.set_xlim(*_compute_axis_limits(raw_x, air_x, gt_x))
    ax_full.set_ylim(*_compute_axis_limits(raw_y, air_y, gt_y))
    ax_full.set_zlim(*_compute_axis_limits(raw_z, air_z, gt_z))

    ax_zoom.set_title('3D zoomed (AirIMU + GT)')
    ax_zoom.set_xlim(*_compute_axis_limits(air_x, gt_x))
    ax_zoom.set_ylim(*_compute_axis_limits(air_y, gt_y))
    ax_zoom.set_zlim(*_compute_axis_limits(air_z, gt_z))
    ax_zoom.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, save_prefix + "_trajectory_3d.png"), dpi=300)
    plt.close()


def box_plot_wrapper(ax, data, edge_color, fill_color, **kwargs):
    bp = ax.boxplot(data, **kwargs)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp


def plot_boxes(folder, input_data, metrics, show_metrics):
    fig, ax = plt.subplots(dpi=300)
    raw_ticks   = [_-0.12 for _ in range(1, len(metrics) + 1)]
    air_ticks   = [_+0.12 for _ in range(1, len(metrics) + 1)]
    label_ticks = [_      for _ in range(1, len(metrics) + 1)]
    
    raw_data    = [input_data[metric + "(raw)"   ] for metric in metrics]
    air_data    = [input_data[metric + "(AirIMU)"] for metric in metrics]
    
    # ax.boxplot(data, patch_artist=True, positions=ticks, widths=.2)
    box_plot_wrapper(ax, raw_data, edge_color="black", fill_color="royalblue", positions=raw_ticks, patch_artist=True, widths=.2)
    box_plot_wrapper(ax, air_data, edge_color="black", fill_color="gold", positions=air_ticks, patch_artist=True, widths=.2)
    ax.set_xticks(label_ticks)
    ax.set_xticklabels(show_metrics)
    
    # Create color patches for legend
    gold_patch = mpatches.Patch(color='gold', label='AirIMU')
    royalblue_patch = mpatches.Patch(color='royalblue', label='Raw')
    ax.legend(handles=[gold_patch, royalblue_patch])
    
    plt.savefig(os.path.join(folder, "Metrics.png"), dpi = 300)
    plt.close()
