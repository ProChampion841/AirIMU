#!/usr/bin/env python3
# convert_blackbird_to_euroc.py
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [r for r in csv.reader(f) if r]

def has_header(row):
    try:
        [float(x) for x in row]
        return False
    except:
        return True

def to_array(rows):
    return np.asarray([[float(x) for x in r] for r in rows], dtype=np.float64)

def convert_ts(t, unit):
    if unit=="ns": return t.astype(np.int64)
    if unit=="us": return (t*1000).astype(np.int64)
    if unit=="ms": return (t*1_000_000).astype(np.int64)
    if unit=="s":  return (t*1e9).astype(np.int64)
    raise ValueError("bad unit")

def parse_axis_map(s):
    lut={"x":0,"y":1,"z":2}
    out=[]
    for tok in s.split(","):
        tok=tok.strip().lower()
        sign=1
        if tok.startswith("-"):
            sign=-1
            tok=tok[1:]
        out.append((lut[tok],sign))
    return out

def apply_axis(v, axis_map):
    spec=parse_axis_map(axis_map)
    out=np.zeros_like(v)
    for i,(src,sgn) in enumerate(spec):
        out[:,i]=sgn*v[:,src]
    return out

def parse_imu(path, unit="us", swap_labeled=True):
    rows=read_csv_rows(path)
    header=None
    if has_header(rows[0]):
        header=[h.strip().lower() for h in rows[0]]
        rows=rows[1:]
    data=to_array(rows)

    t_ns=convert_ts(data[:,0],unit)

    if header and "acc_x" in header and "gyro_x" in header:
        ax,ay,az=[header.index(k) for k in ["acc_x","acc_y","acc_z"]]
        gx,gy,gz=[header.index(k) for k in ["gyro_x","gyro_y","gyro_z"]]
        acc=data[:,[ax,ay,az]]
        gyro=data[:,[gx,gy,gz]]
        if swap_labeled:
            acc,gyro=gyro,acc
    else:
        acc=data[:,1:4]
        gyro=data[:,4:7]

    return t_ns,acc,gyro

def parse_gt(path,unit="us"):
    rows=read_csv_rows(path)
    if has_header(rows[0]):
        rows=rows[1:]
    data=to_array(rows)
    t_ns=convert_ts(data[:,0],unit)
    pos=data[:,1:4]
    quat=data[:,4:8]
    return t_ns,pos,quat

def ned_to_enu_pos(p):
    out=np.zeros_like(p)
    out[:,0]=p[:,1]
    out[:,1]=p[:,0]
    out[:,2]=-p[:,2]
    return out

def normalize_q(q):
    return q/np.linalg.norm(q,axis=1,keepdims=True)

def vel_from_pos(t,p):
    v=np.zeros_like(p)
    dt=(t[1:]-t[:-1])/1e9
    dt=np.maximum(dt,1e-12)
    v[1:]=(p[1:]-p[:-1])/dt[:,None]
    v[0]=v[1]
    return v

def write_csv(path,header,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(header)
        w.writerows(data.tolist())

def convert_sequence(src,dst,ts_unit="us",axis_map="x,y,z"):
    t_imu,acc,gyro=parse_imu(src/"imu_data.csv",ts_unit)
    t_gt,pos,quat=parse_gt(src/"groundTruthPoses.csv",ts_unit)

    acc=apply_axis(acc,axis_map)
    gyro=apply_axis(gyro,axis_map)

    pos=ned_to_enu_pos(pos)
    quat=normalize_q(quat)

    vel=vel_from_pos(t_gt,pos)
    bias=np.zeros((len(pos),6))

    imu_rows=np.column_stack([t_imu,gyro,acc])
    gt_rows=np.column_stack([t_gt,pos,quat,vel,bias])

    write_csv(dst/"mav0/imu0/data.csv",
        ["#timestamp","w_x","w_y","w_z","a_x","a_y","a_z"],
        imu_rows)

    write_csv(dst/"mav0/state_groundtruth_estimate0/data.csv",
        ["#timestamp","px","py","pz","qw","qx","qy","qz",
         "vx","vy","vz","bgx","bgy","bgz","bax","bay","baz"],
        gt_rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input_root",required=True)
    ap.add_argument("--output_root",required=True)
    ap.add_argument("--imu_axis_map",default="x,y,z")
    ap.add_argument("--timestamp_unit",default="us")
    args=ap.parse_args()

    in_root=Path(args.input_root)
    out_root=Path(args.output_root)

    for seq in sorted([p for p in in_root.iterdir() if p.is_dir()]):
        print("Converting",seq.name)
        convert_sequence(seq,out_root/seq.name,args.timestamp_unit,args.imu_axis_map)

if __name__=="__main__":
    main()
