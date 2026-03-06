#!/usr/bin/env python3
"""Convert Blackbird sequences to EuRoC-like folder/file format.

Input per sequence (Blackbird):
  - imu_data.csv
      # timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  - groundTruthPoses.csv
      TimeStamp, x, y, z, qw, qx, qy, qz
      (x, y, z are NED world coordinates)

Output per sequence (EuRoC-like):
  - mav0/imu0/data.csv
      timestamp [ns], w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z
  - mav0/state_groundtruth_estimate0/data.csv
      timestamp [ns], p_RS_R_x, p_RS_R_y, p_RS_R_z,
      q_RS_w, q_RS_x, q_RS_y, q_RS_z,
      v_RS_R_x, v_RS_R_y, v_RS_R_z,
      b_w_RS_S_x, b_w_RS_S_y, b_w_RS_S_z,
      b_a_RS_S_x, b_a_RS_S_y, b_a_RS_S_z

Notes:
  - By default, this script converts GT world frame NED -> ENU for position + orientation.
  - IMU measurements are kept as-is (sensor/body frame), only reordered to EuRoC column order.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np


def _read_csv(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    return rows


def _to_float_rows(rows):
    return np.asarray([[float(x.strip()) for x in r] for r in rows], dtype=np.float64)


def _parse_imu(imu_path: Path):
    rows = _read_csv(imu_path)
    header = [h.strip().lower() for h in rows[0]]
    if header and header[0].startswith("#"):
        header[0] = header[0][1:].strip()

    expected = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    if header != expected:
        raise ValueError(f"Unexpected IMU header in {imu_path}: {header} != {expected}")

    data = _to_float_rows(rows[1:])
    if data.shape[1] < 7:
        raise ValueError(f"IMU file must have >=7 columns: {imu_path}")

    t = data[:, 0].astype(np.int64)
    acc = data[:, 1:4]
    gyro = data[:, 4:7]
    return t, acc, gyro


def _parse_gt(gt_path: Path):
    rows = _read_csv(gt_path)

    # Blackbird groundTruthPoses.csv is commonly provided WITHOUT a header.
    # Expected column order:
    #   TimeStamp, x, y, z, qw, qx, qy, qz
    # We still support an optional header if present.
    first_tokens = [x.strip().lower() for x in rows[0]]

    has_header = False
    try:
        [float(x) for x in first_tokens]
    except ValueError:
        has_header = True

    if has_header:
        expected = ["timestamp", "x", "y", "z", "qw", "qx", "qy", "qz"]
        if first_tokens != expected:
            raise ValueError(f"Unexpected GT header in {gt_path}: {first_tokens} != {expected}")
        numeric_rows = rows[1:]
    else:
        numeric_rows = rows

    data = _to_float_rows(numeric_rows)
    if data.shape[1] < 8:
        raise ValueError(f"GT file must have >=8 columns: {gt_path}")

    t = data[:, 0].astype(np.int64)
    pos_ned = data[:, 1:4]
    quat_wxyz_ned = data[:, 4:8]
    return t, pos_ned, quat_wxyz_ned


def _quat_normalize(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return q / n


def _quat_mul(q1, q2):
    """Hamilton product for wxyz quaternions."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    out = np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)
    return _quat_normalize(out)


def _rotation_matrix_to_quat_wxyz(R):
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return _quat_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))


def _compute_velocity_world_ns(t_ns, pos_world):
    vel = np.zeros_like(pos_world)
    if len(t_ns) > 1:
        dt = np.maximum((t_ns[1:] - t_ns[:-1]).astype(np.float64) / 1e9, 1e-9)
        vel[1:] = (pos_world[1:] - pos_world[:-1]) / dt[:, None]
        vel[0] = vel[1]
    return vel


def _convert_ned_to_enu_position(pos_ned):
    # NED [N, E, D] -> ENU [E, N, U]
    pos_enu = np.empty_like(pos_ned)
    pos_enu[:, 0] = pos_ned[:, 1]
    pos_enu[:, 1] = pos_ned[:, 0]
    pos_enu[:, 2] = -pos_ned[:, 2]
    return pos_enu


def _convert_ned_to_enu_quat(quat_wxyz_ned):
    # R_enu = C * R_ned, with C mapping NED basis into ENU basis.
    C = np.array([[0.0, 1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0]], dtype=np.float64)
    q_c = _rotation_matrix_to_quat_wxyz(C)
    q_c_batch = np.repeat(q_c[None, :], len(quat_wxyz_ned), axis=0)
    return _quat_mul(q_c_batch, _quat_normalize(quat_wxyz_ned))


def _ensure_monotonic_unique(t, *arrays):
    idx = np.argsort(t)
    t = t[idx]
    arrays = [a[idx] for a in arrays]
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 0
    return (t[keep], *[a[keep] for a in arrays])


def _write_csv(path: Path, header, data_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)


def convert_sequence(seq_in: Path, seq_out: Path, ned_to_enu: bool):
    imu_t, imu_acc, imu_gyro = _parse_imu(seq_in / "imu_data.csv")
    gt_t, pos_ned, quat_ned = _parse_gt(seq_in / "groundTruthPoses.csv")

    imu_t, imu_acc, imu_gyro = _ensure_monotonic_unique(imu_t, imu_acc, imu_gyro)
    gt_t, pos_ned, quat_ned = _ensure_monotonic_unique(gt_t, pos_ned, quat_ned)

    if ned_to_enu:
        pos_world = _convert_ned_to_enu_position(pos_ned)
        quat_world = _convert_ned_to_enu_quat(quat_ned)
    else:
        pos_world = pos_ned
        quat_world = quat_ned

    vel_world = _compute_velocity_world_ns(gt_t, pos_world)
    zeros_bias = np.zeros((len(gt_t), 6), dtype=np.float64)

    imu_rows = np.column_stack([imu_t, imu_gyro, imu_acc])
    gt_rows = np.column_stack([gt_t, pos_world, quat_world, vel_world, zeros_bias])

    _write_csv(
        seq_out / "mav0" / "imu0" / "data.csv",
        [
            "#timestamp [ns]",
            "w_RS_S_x [rad s^-1]",
            "w_RS_S_y [rad s^-1]",
            "w_RS_S_z [rad s^-1]",
            "a_RS_S_x [m s^-2]",
            "a_RS_S_y [m s^-2]",
            "a_RS_S_z [m s^-2]",
        ],
        imu_rows,
    )

    _write_csv(
        seq_out / "mav0" / "state_groundtruth_estimate0" / "data.csv",
        [
            "#timestamp",
            "p_RS_R_x [m]",
            "p_RS_R_y [m]",
            "p_RS_R_z [m]",
            "q_RS_w []",
            "q_RS_x []",
            "q_RS_y []",
            "q_RS_z []",
            "v_RS_R_x [m s^-1]",
            "v_RS_R_y [m s^-1]",
            "v_RS_R_z [m s^-1]",
            "b_w_RS_S_x [rad s^-1]",
            "b_w_RS_S_y [rad s^-1]",
            "b_w_RS_S_z [rad s^-1]",
            "b_a_RS_S_x [m s^-2]",
            "b_a_RS_S_y [m s^-2]",
            "b_a_RS_S_z [m s^-2]",
        ],
        gt_rows,
    )


def main():
    p = argparse.ArgumentParser(description="Convert Blackbird sequences to EuRoC-like format")
    p.add_argument("--input_root", required=True, help="Root containing Blackbird sequence folders (seq_00, ...)")
    p.add_argument("--output_root", required=True, help="Output root for EuRoC-like folders")
    p.add_argument("--sequences", nargs="*", default=None, help="Sequence names to convert. Default: all dirs under input_root")
    p.add_argument("--keep_ned", action="store_true", help="Keep GT in NED (do NOT convert to ENU)")
    args = p.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)

    if args.sequences:
        seqs = args.sequences
    else:
        seqs = sorted([p.name for p in in_root.iterdir() if p.is_dir()])

    if not seqs:
        raise ValueError("No sequences found to convert.")

    for seq in seqs:
        src = in_root / seq
        dst = out_root / seq
        print(f"Converting {src} -> {dst}")
        convert_sequence(src, dst, ned_to_enu=(not args.keep_ned))

    print("Done.")


if __name__ == "__main__":
    main()