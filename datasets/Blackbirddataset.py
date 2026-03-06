import os

import numpy as np
import pypose as pp
import torch

from utils import qinterp
from .dataset import Sequence


class Blackbird(Sequence):
    """Loader for Blackbird dataset sequences.

    Expected per-sequence files:
      - imu_data.csv with columns for timestamp + (acc, gyro)
      - groundTruthPoses.csv with timestamp, position, quaternion

    Notes:
      - Supports CSVs with or without a header row.
      - Supports headers that start with '#'.
    """

    def __init__(self, data_root, data_name, intepolate=True, calib=False, glob_coord=False, quat_order="wxyz", **kwargs):
        super(Blackbird, self).__init__()
        self.data = {}
        self.quat_order = str(quat_order).lower()

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
        raw_time = np.asarray(raw_time, dtype=np.float64)
        if raw_time.size <= 1:
            return raw_time

        # Decide the input unit by checking the median delta between samples.
        med_dt = np.median(np.diff(raw_time))
        if med_dt <= 0:
            return raw_time

        sec_dt = [
            med_dt,         # already seconds
            med_dt / 1e3,   # milliseconds
            med_dt / 1e6,   # microseconds
            med_dt / 1e9,   # nanoseconds
        ]
        scales = [1.0, 1e3, 1e6, 1e9]
        target_dt = 1e-2
        best_idx = int(np.argmin([abs(np.log10(max(dt, 1e-12) / target_dt)) for dt in sec_dt]))
        return raw_time / scales[best_idx]

    @staticmethod
    def _find_idx(headers, aliases):
        for alias in aliases:
            if alias in headers:
                return headers.index(alias)
        raise ValueError(f"Cannot find any of aliases {aliases} in header: {headers}")

    @staticmethod
    def _is_numeric_row(tokens):
        if not tokens:
            return False
        try:
            [float(tok) for tok in tokens]
            return True
        except ValueError:
            return False

    @classmethod
    def _load_csv_with_optional_header(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()

        first_line_clean = first_line[1:] if first_line.startswith("#") else first_line
        first_tokens = [tok.strip() for tok in first_line_clean.split(",") if tok.strip() != ""]
        has_header = not cls._is_numeric_row(first_tokens)

        headers = None
        if has_header:
            headers = [h.strip().lower() for h in first_tokens]
            data = np.loadtxt(file_path, delimiter=",", comments="#", skiprows=1, dtype=np.float64)
        else:
            data = np.loadtxt(file_path, delimiter=",", comments="#", dtype=np.float64)

        if data.ndim == 1:
            data = data[None, :]

        return headers, data

    @staticmethod
    def _sort_and_unique_time(time_arr, *value_arrs):
        order = np.argsort(time_arr)
        time_sorted = np.asarray(time_arr)[order]
        values_sorted = [np.asarray(arr)[order] for arr in value_arrs]

        # Keep strictly increasing timestamps.
        keep = np.ones_like(time_sorted, dtype=bool)
        keep[1:] = np.diff(time_sorted) > 0
        time_sorted = time_sorted[keep]
        values_sorted = [arr[keep] for arr in values_sorted]

        return (time_sorted, *values_sorted)

    def load_imu(self, folder):
        imu_path = os.path.join(folder, "imu_data.csv")
        headers, data = self._load_csv_with_optional_header(imu_path)

        if headers is None:
            if data.shape[1] < 7:
                raise ValueError(f"Expected at least 7 columns in {imu_path}, got {data.shape[1]}.")
            t_idx, ax_idx, ay_idx, az_idx, gx_idx, gy_idx, gz_idx = 0, 1, 2, 3, 4, 5, 6
        else:
            t_idx = self._find_idx(headers, ["timestamp", "time", "t"])
            ax_idx = self._find_idx(headers, ["acc_x", "ax", "a_x"])
            ay_idx = self._find_idx(headers, ["acc_y", "ay", "a_y"])
            az_idx = self._find_idx(headers, ["acc_z", "az", "a_z"])
            gx_idx = self._find_idx(headers, ["gyro_x", "wx", "gyr_x", "w_x"])
            gy_idx = self._find_idx(headers, ["gyro_y", "wy", "gyr_y", "w_y"])
            gz_idx = self._find_idx(headers, ["gyro_z", "wz", "gyr_z", "w_z"])

        raw_time = self._normalize_time(data[:, t_idx])
        acc = data[:, [ax_idx, ay_idx, az_idx]]
        gyro = data[:, [gx_idx, gy_idx, gz_idx]]

        raw_time, acc, gyro = self._sort_and_unique_time(raw_time, acc, gyro)

        self.data["time"] = raw_time
        self.data["acc"] = acc
        self.data["gyro"] = gyro

    def load_gt(self, folder):
        gt_path = os.path.join(folder, "groundTruthPoses.csv")
        headers, data = self._load_csv_with_optional_header(gt_path)

        if data.shape[1] < 8:
            raise ValueError(f"Expected at least 8 columns in {gt_path}, got {data.shape[1]}.")

        if headers is None:
            t_idx = 0
            x_idx, y_idx, z_idx = 1, 2, 3
            if self.quat_order == "xyzw":
                qx_idx, qy_idx, qz_idx, qw_idx = 4, 5, 6, 7
            else:
                qw_idx, qx_idx, qy_idx, qz_idx = 4, 5, 6, 7
        else:
            t_idx = self._find_idx(headers, ["timestamp", "time", "t"])
            x_idx = self._find_idx(headers, ["x", "p_x", "pos_x", "position_x"])
            y_idx = self._find_idx(headers, ["y", "p_y", "pos_y", "position_y"])
            z_idx = self._find_idx(headers, ["z", "p_z", "pos_z", "position_z"])

            if "qw" in headers:
                qw_idx = self._find_idx(headers, ["qw", "q_w", "quat_w", "w"])
                qx_idx = self._find_idx(headers, ["qx", "q_x", "quat_x"])
                qy_idx = self._find_idx(headers, ["qy", "q_y", "quat_y"])
                qz_idx = self._find_idx(headers, ["qz", "q_z", "quat_z"])
            else:
                # Fallback: some files may expose x/y/z/w style naming.
                qx_idx = self._find_idx(headers, ["qx", "q_x", "quat_x", "x_q"])
                qy_idx = self._find_idx(headers, ["qy", "q_y", "quat_y", "y_q"])
                qz_idx = self._find_idx(headers, ["qz", "q_z", "quat_z", "z_q"])
                qw_idx = self._find_idx(headers, ["qw", "q_w", "quat_w", "w_q", "w"])

        raw_time = self._normalize_time(data[:, t_idx])
        pos = data[:, [x_idx, y_idx, z_idx]]
        quat_wxyz = data[:, [qw_idx, qx_idx, qy_idx, qz_idx]]

        raw_time, pos, quat_wxyz = self._sort_and_unique_time(raw_time, pos, quat_wxyz)

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