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
                            save_folder=None, mask=None, file_name="state_error_compare.png",  plot_title=None):
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
    fig.suptitle(plot_title or "Integration error vs AirIMU Integration error")
    
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
    # plt.show()


def _set_2d_equal_aspect(ax, xlim, ylim):
    """Keep one unit on X equal to one unit on Y without changing data bounds."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')

def _set_3d_equal_aspect(ax, xlim, ylim, zlim):
    """Use true data spans for 3D aspect so units are consistent without over-padding."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    x_span = max(float(xlim[1] - xlim[0]), 1e-6)
    y_span = max(float(ylim[1] - ylim[0]), 1e-6)
    z_span = max(float(zlim[1] - zlim[0]), 1e-6)
    ax.set_box_aspect((x_span, y_span, z_span))


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
    # plt.show()


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
    full_xlim = _compute_axis_limits(raw_x, air_x, gt_x)
    full_ylim = _compute_axis_limits(raw_y, air_y, gt_y)
    _set_2d_equal_aspect(axes[0], full_xlim, full_ylim)
    axes[0].legend()

    # Zoomed view focused on AirIMU and GT overlap.
    axes[1].plot(raw_x, raw_y, label="Raw", linewidth=1.0, alpha=0.25)
    axes[1].plot(air_x, air_y, label="AirIMU", linewidth=1.4)
    axes[1].plot(gt_x, gt_y, label="Ground Truth", linewidth=1.4)
    axes[1].set_title("Zoomed (AirIMU + GT region)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].grid(True)
    zoom_xlim = _compute_axis_limits(air_x, gt_x)
    zoom_ylim = _compute_axis_limits(air_y, gt_y)
    _set_2d_equal_aspect(axes[1], zoom_xlim, zoom_ylim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def _truncate_to_common_length(*arrays):
    """Trim arrays to common prefix length for apples-to-apples plotting."""
    min_len = min(len(np.asarray(arr).reshape(-1)) for arr in arrays)
    return [np.asarray(arr).reshape(-1)[:min_len] for arr in arrays], min_len

def visualize_trajectory(save_prefix, save_folder, outstate, infstate):
    gt_source = infstate if "poses_gt" in infstate else outstate

    gt_x, gt_y, gt_z = torch.split(gt_source["poses_gt"][0].cpu(), 1, dim=1)
    raw_x, raw_y, raw_z = torch.split(outstate["poses"][0].cpu(), 1, dim=1)
    air_x, air_y, air_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)

    gt_x, gt_y, gt_z = _to_numpy_1d(gt_x), _to_numpy_1d(gt_y), _to_numpy_1d(gt_z)
    raw_x, raw_y, raw_z = _to_numpy_1d(raw_x), _to_numpy_1d(raw_y), _to_numpy_1d(raw_z)
    air_x, air_y, air_z = _to_numpy_1d(air_x), _to_numpy_1d(air_y), _to_numpy_1d(air_z)
    (raw_x, raw_y, raw_z, air_x, air_y, air_z, gt_x, gt_y, gt_z), common_len = _truncate_to_common_length(
        raw_x, raw_y, raw_z, air_x, air_y, air_z, gt_x, gt_y, gt_z
    )

    # Match the position-error definition used in training/evaluation losses,
    # i.e., compare trajectories point-wise in the same reference frame.
    # We anchor all curves to the first GT point to remove constant offsets.
    ref_x, ref_y, ref_z = gt_x[0], gt_y[0], gt_z[0]
    raw_x, raw_y, raw_z = raw_x - ref_x, raw_y - ref_y, raw_z - ref_z
    air_x, air_y, air_z = air_x - ref_x, air_y - ref_y, air_z - ref_z
    gt_x, gt_y, gt_z = gt_x - ref_x, gt_y - ref_y, gt_z - ref_z
    
    
    print(
        f"[visualize_trajectory] {save_prefix}: using {common_len} synchronized points | "
        f"Raw x/y/z ranges=({raw_x.min():.2f},{raw_x.max():.2f}) / ({raw_y.min():.2f},{raw_y.max():.2f}) / ({raw_z.min():.2f},{raw_z.max():.2f}) | "
        f"AirIMU x/y/z ranges=({air_x.min():.2f},{air_x.max():.2f}) / ({air_y.min():.2f},{air_y.max():.2f}) / ({air_z.min():.2f},{air_z.max():.2f}) | "
        f"GT x/y/z ranges=({gt_x.min():.2f},{gt_x.max():.2f}) / ({gt_y.min():.2f},{gt_y.max():.2f}) / ({gt_z.min():.2f},{gt_z.max():.2f})"
    )
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
    full_xlim = _compute_axis_limits(raw_x, air_x, gt_x)
    full_ylim = _compute_axis_limits(raw_y, air_y, gt_y)
    full_zlim = _compute_axis_limits(raw_z, air_z, gt_z)
    _set_3d_equal_aspect(ax_full, full_xlim, full_ylim, full_zlim)
    
    ax_zoom.set_title('3D zoomed (AirIMU + GT)')
    zoom_xlim = _compute_axis_limits(air_x, gt_x)
    zoom_ylim = _compute_axis_limits(air_y, gt_y)
    zoom_zlim = _compute_axis_limits(air_z, gt_z)
    _set_3d_equal_aspect(ax_zoom, zoom_xlim, zoom_ylim, zoom_zlim)
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