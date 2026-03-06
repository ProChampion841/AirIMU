import os

import numpy as np
import pypose as pp
import torch

from utils import qinterp
from .dataset import Sequence


class Blackbird(Sequence):
    """Loader for Blackbird dataset sequences.

    Expected per-sequence files:
      - imu_data.csv with header:
        # timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
      - groundTruthPoses.csv without header
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

            for key in ["gt_time", "pos", "quat", "velocity"]:
                self.data[key] = self.data[key][idx_start_gt:idx_end_gt]

            for key in ["time", "acc", "gyro"]:
                self.data[key] = self.data[key][idx_start_imu:idx_end_imu]

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
    def _normalize_time(raw_time):
        if raw_time.max() > 1e14:
            return raw_time / 1e9
        if raw_time.max() > 1e11:
            return raw_time / 1e6
        return raw_time

    @staticmethod
    def _parse_imu_header(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        if first_line.startswith("#"):
            header = first_line[1:]
        else:
            header = first_line

        headers = [h.strip().lower() for h in header.split(",")]
        return headers

    @staticmethod
    def _find_idx(headers, aliases):
        for alias in aliases:
            if alias in headers:
                return headers.index(alias)
        raise ValueError(f"Cannot find any of aliases {aliases} in header: {headers}")

    def load_imu(self, folder):
        imu_path = os.path.join(folder, "imu_data.csv")
        headers = self._parse_imu_header(imu_path)

        data = np.loadtxt(imu_path, delimiter=",", comments="#", dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]

        t_idx = self._find_idx(headers, ["timestamp", "time", "t"])
        ax_idx = self._find_idx(headers, ["acc_x", "ax", "a_x"])
        ay_idx = self._find_idx(headers, ["acc_y", "ay", "a_y"])
        az_idx = self._find_idx(headers, ["acc_z", "az", "a_z"])
        gx_idx = self._find_idx(headers, ["gyro_x", "wx", "gyr_x", "w_x"])
        gy_idx = self._find_idx(headers, ["gyro_y", "wy", "gyr_y", "w_y"])
        gz_idx = self._find_idx(headers, ["gyro_z", "wz", "gyr_z", "w_z"])

        raw_time = self._normalize_time(data[:, t_idx])

        self.data["time"] = raw_time
        self.data["acc"] = data[:, [ax_idx, ay_idx, az_idx]]
        self.data["gyro"] = data[:, [gx_idx, gy_idx, gz_idx]]

    def load_gt(self, folder):
        gt_path = os.path.join(folder, "groundTruthPoses.csv")
        data = np.loadtxt(gt_path, delimiter=",", dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]

        # Blackbird groundTruthPoses.csv has no header. Use standard 8-column layout:
        # timestamp, x, y, z, qw, qx, qy, qz
        if data.shape[1] < 8:
            raise ValueError(f"Expected at least 8 columns in {gt_path}, got {data.shape[1]}.")

        raw_time = self._normalize_time(data[:, 0])
        pos = data[:, 1:4]
        quat_wxyz = data[:, 4:8]

        vel = np.zeros_like(pos)
        delta_t = np.maximum(np.diff(raw_time), 1e-6)
        vel[1:] = (pos[1:] - pos[:-1]) / delta_t[:, None]
        if len(vel) > 1:
            vel[0] = vel[1]

        self.data["gt_time"] = raw_time
        self.data["pos"] = pos
        self.data["quat"] = quat_wxyz
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