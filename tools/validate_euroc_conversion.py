#!/usr/bin/env python3
# validate_euroc_conversion.py
import argparse, csv
from pathlib import Path
import numpy as np

def read(path):
    with path.open("r") as f:
        rows=[r for r in csv.reader(f) if r]
    if not rows: raise RuntimeError("empty csv")
    if not rows[0][0].replace(".","").isdigit():
        rows=rows[1:]
    return np.asarray([[float(x) for x in r] for r in rows])

def check(seq):
    imu=read(seq/"mav0/imu0/data.csv")
    gt=read(seq/"mav0/state_groundtruth_estimate0/data.csv")

    t_imu=imu[:,0]
    gyro=imu[:,1:4]
    acc=imu[:,4:7]

    t_gt=gt[:,0]
    pos=gt[:,1:4]
    vel=gt[:,8:11]

    print("IMU freq:",1e9/np.median(np.diff(t_imu)))
    print("GT freq:",1e9/np.median(np.diff(t_gt)))

    gnorm=np.linalg.norm(gyro,axis=1)
    anorm=np.linalg.norm(acc,axis=1)

    print("gyro median:",np.median(gnorm))
    print("acc median:",np.median(anorm))

    if anorm.mean()<2 and gnorm.mean()>3:
        print("FAIL: accel/gyro likely swapped")

    v_est=np.zeros_like(pos)
    dt=(t_gt[1:]-t_gt[:-1])/1e9
    v_est[1:]=(pos[1:]-pos[:-1])/dt[:,None]
    err=np.linalg.norm(v_est-vel,axis=1)
    print("velocity error median:",np.median(err))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sequence",required=True)
    args=ap.parse_args()
    check(Path(args.sequence))

if __name__=="__main__":
    main()
