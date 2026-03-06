import csv
import os

import numpy as np
import pypose as pp
import torch

from utils import qinterp
from .dataset import Sequence


class Blackbird(Sequence):
    """Loader for Blackbird-style sequences.

    Expected per-sequence files:
      - imu_data.csv
      - groundTruthPoses.csv

    The loader tries to auto-detect common column names used by Blackbird exports.
    """

    def __init__(self, data_root, data_name, intepolate=True, calib=False, glob_coord=False, **kwargs):
        super(Blackbird, self).__init__()
        self.data = {}

        data_path = os.path.join(data_root, data_name)
        self.load_imu(data_path)
        self.load_gt(data_path)

        if intepolate:
            t_start = np.max([self.data["gt_time"][0], self.data["time"][0]])
            t_end = np.min([self.data["gt_time"][-1], self.data["time"][-1]])

            idx_start_imu = np.searchsorted(self.data["time"], t_start)
            idx_start_gt = np.searchsorted(self.data["gt_time"], t_start)

            idx_end_imu = np.searchsorted(self.data["time"], t_end, "right")
            idx_end_gt = np.searchsorted(self.data["gt_time"], t_end, "right")

            for k in ["gt_time", "pos", "quat", "velocity"]:
                self.data[k] = self.data[k][idx_start_gt:idx_end_gt]

            for k in ["time", "acc", "gyro"]:
                self.data[k] = self.data[k][idx_start_imu:idx_end_imu]

            self.data["gt_orientation"] = self.interp_rot(self.data["time"], self.data["gt_time"], self.data["quat"])
            self.data["gt_translation"] = self.interp_xyz(self.data["time"], self.data["gt_time"], self.data["pos"])
            self.data["velocity"] = self.interp_xyz(self.data["time"], self.data["gt_time"], self.data["velocity"])
        else:
            quat = torch.tensor(self.data["quat"])  # wxyz
            rot = torch.zeros_like(quat)
            rot[:, :3] = quat[:, 1:]
            rot[:, 3] = quat[:, 0]
            self.data["gt_orientation"] = pp.SO3(rot)
            self.data["gt_translation"] = torch.tensor(self.data["pos"])

        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

        self.data["gyro"] = torch.tensor(self.data["gyro"])
        self.data["acc"] = torch.tensor(self.data["acc"])

        if calib:
            self.data["gyro"] = self.data["gyro"] - self.data["gyro"][0]
            self.data["acc"] = self.data["acc"] - self.data["acc"][0]

        if glob_coord:
            self.data["gyro"] = self.data["gt_orientation"] * self.data["gyro"]
            self.data["acc"] = self.data["gt_orientation"] * self.data["acc"]

        print("loaded:", data_path, "calib:", calib, "interpolate:", intepolate)

    def get_length(self):
        return self.data["time"].shape[0]

    @staticmethod
    def _find_idx(headers, aliases):
        lowered = [h.strip().lower() for h in headers]
        for alias in aliases:
            if alias in lowered:
                return lowered.index(alias)
        raise ValueError(f"Cannot find any of aliases {aliases} in header: {headers}")

    def _load_csv(self, file_path):
        with open(file_path, "r", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = [[float(v) for v in row] for row in reader if len(row) > 0]
        return headers, np.asarray(rows, dtype=np.float64)

    def load_imu(self, folder):
        headers, data = self._load_csv(os.path.join(folder, "imu_data.csv"))

        t_idx = self._find_idx(headers, ["timestamp", "t", "time", "timestamp(ns)", "timestamp_us"])

        gx_idx = self._find_idx(headers, ["wx", "gyro_x", "gyr_x", "angular_velocity_x", "w_x"])
        gy_idx = self._find_idx(headers, ["wy", "gyro_y", "gyr_y", "angular_velocity_y", "w_y"])
        gz_idx = self._find_idx(headers, ["wz", "gyro_z", "gyr_z", "angular_velocity_z", "w_z"])

        ax_idx = self._find_idx(headers, ["ax", "acc_x", "linear_acceleration_x", "a_x"])
        ay_idx = self._find_idx(headers, ["ay", "acc_y", "linear_acceleration_y", "a_y"])
        az_idx = self._find_idx(headers, ["az", "acc_z", "linear_acceleration_z", "a_z"])

        raw_time = data[:, t_idx]
        # Normalize to seconds if timestamp appears in ns/us.
        if raw_time.max() > 1e14:
            raw_time = raw_time / 1e9
        elif raw_time.max() > 1e11:
            raw_time = raw_time / 1e6

        self.data["time"] = raw_time
        self.data["gyro"] = data[:, [gx_idx, gy_idx, gz_idx]]
        self.data["acc"] = data[:, [ax_idx, ay_idx, az_idx]]

    def load_gt(self, folder):
        headers, data = self._load_csv(os.path.join(folder, "groundTruthPoses.csv"))

        t_idx = self._find_idx(headers, ["timestamp", "t", "time", "timestamp(ns)", "timestamp_us"])

        px_idx = self._find_idx(headers, ["x", "px", "position_x", "p_x"])
        py_idx = self._find_idx(headers, ["y", "py", "position_y", "p_y"])
        pz_idx = self._find_idx(headers, ["z", "pz", "position_z", "p_z"])

        qx_idx = self._find_idx(headers, ["qx", "quat_x", "orientation_x"])
        qy_idx = self._find_idx(headers, ["qy", "quat_y", "orientation_y"])
        qz_idx = self._find_idx(headers, ["qz", "quat_z", "orientation_z"])
        qw_idx = self._find_idx(headers, ["qw", "quat_w", "orientation_w"])
        quat = data[:, [qw_idx, qx_idx, qy_idx, qz_idx]]

        raw_time = data[:, t_idx]
        if raw_time.max() > 1e14:
            raw_time = raw_time / 1e9
        elif raw_time.max() > 1e11:
            raw_time = raw_time / 1e6

        pos = data[:, [px_idx, py_idx, pz_idx]]
        vel = np.zeros_like(pos)
        delta_t = np.maximum(np.diff(raw_time), 1e-6)
        vel[1:] = (pos[1:] - pos[:-1]) / delta_t[:, None]
        if len(vel) > 1:
            vel[0] = vel[1]

        self.data["gt_time"] = raw_time
        self.data["pos"] = pos
        self.data["quat"] = quat
        self.data["velocity"] = vel

    def interp_rot(self, time, gt_time, quat):
        imu_dt = torch.tensor(time - gt_time[0])
        gt_dt = torch.tensor(gt_time - gt_time[0])

        quat = torch.tensor(quat)
        quat = qinterp(quat, gt_dt, imu_dt).double()
        rot = torch.zeros_like(quat)
        rot[:, 3] = quat[:, 0]
        rot[:, :3] = quat[:, 1:]
        return pp.SO3(rot)

    def interp_xyz(self, time, gt_time, xyz):
        x = np.interp(time, xp=gt_time, fp=xyz[:, 0])
        y = np.interp(time, xp=gt_time, fp=xyz[:, 1])
        z = np.interp(time, xp=gt_time, fp=xyz[:, 2])
        return torch.tensor(np.stack([x, y, z]).transpose())