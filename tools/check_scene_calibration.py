"""Run a reviewable development check; never scores held-out violation clips."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

if __name__ == "__main__":
    import torch
    from traffic_edge.intersection_assist import analyze_intersection_video
    parser=argparse.ArgumentParser()
    parser.add_argument("--video",default="data/real_examples/zenodo_video4.mp4")
    parser.add_argument("--time",type=float,default=3)
    parser.add_argument("--seconds",type=float,default=20)
    parser.add_argument("--output",default="runs/calibration-development/day")
    args=parser.parse_args()
    torch.set_num_threads(4)
    output=ROOT/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    print("Analyzing development video",flush=True)
    result=analyze_intersection_video(ROOT/args.video,str(ROOT/"yolov8n.pt"),.15,960,args.seconds,args.time)
    preview=result.pop("scene_preview")
    output.with_suffix(".jpg").write_bytes(base64.b64decode(preview.split(",",1)[1]))
    output.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"lanes":len(result["lane_proposals"]),"signals":len(result["traffic_light_proposals"]),
                      "marking_segments":len(result["marking_boundaries"]),"arrows":result["road_marking_proposals"],
                      "seconds":result["processing_seconds"],"warnings":result["warnings"]},indent=2),flush=True)
