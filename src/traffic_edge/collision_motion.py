"""Offline motion-window proposals inspired by ACCIDENT's optical-flow baseline.

https://github.com/accidentbench/ACCIDENT/tree/main/baselines/heuristic
Motion windows identify intervals for review. Optical flow supplies search
regions independently of detector-box overlap; collision decisions are made separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[2]


def propose(path:Path,target_fps:float=5.,maximum_windows:int=4,start_seconds:float=0.,end_seconds:float|None=None)->dict:
    capture=cv2.VideoCapture(str(path))
    rows=[];previous=None;previous_grid=None;next_time=start_seconds;index=0;rejected=0;quality_intervals=[]
    try:
        if not capture.isOpened():raise ValueError(f'Cannot open {path}')
        fps=capture.get(cv2.CAP_PROP_FPS)
        if fps<=0:raise ValueError('Missing source timing')
        duration=capture.get(cv2.CAP_PROP_FRAME_COUNT)/fps
        index=max(0,int(start_seconds*fps))
        if index:capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        while True:
            ok,frame=capture.read()
            if not ok:break
            t=index/fps;index+=1
            if end_seconds is not None and t>end_seconds:break
            if t+1e-6<next_time:continue
            next_time+=1/target_fps
            h,w=frame.shape[:2];height=round(h*640/w)
            grey=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(640,height))
            # Exclude source marks and player chrome; identical masks for controls.
            mask=np.ones(grey.shape,dtype=bool)
            for l,a,r,b in ((0,0,.3,.08),(0,.88,.35,1),(.85,.92,1,1),(0,.965,1,1)):
                mask[int(a*height):int(b*height),int(l*640):int(r*640)]=False
            if previous is None:previous=grey;continue
            flow=cv2.calcOpticalFlowFarneback(previous,grey,None,.5,3,15,3,5,1.2,0)
            background=np.median(flow[mask],axis=0)
            photometric=np.mean(np.abs(grey.astype(float)-previous.astype(float))[mask])
            previous=grey
            if np.linalg.norm(background)>3 or photometric>45:
                rejected+=1;quality_intervals.append([max(start_seconds,t-1/target_fps),t]);previous_grid=None;continue
            magnitude=np.linalg.norm(flow-background,axis=2)
            grid=np.zeros((6,10),dtype=float)
            for row in range(6):
                for column in range(10):
                    y1,y2=round(row*height/6),round((row+1)*height/6)
                    x1,x2=column*64,(column+1)*64
                    values=magnitude[y1:y2,x1:x2][mask[y1:y2,x1:x2]]
                    grid[row,column]=float(np.percentile(values,90)) if values.size else 0.
            if previous_grid is not None:
                change=np.abs(grid-previous_grid)
                row,column=np.unravel_index(np.argmax(change),change.shape)
                rows.append({'time':round(t,6),'motion_change':float(np.percentile(change,95)),
                             'region':[max(0,(column-1)/10),max(0,(row-1)/6),min(1,(column+2)/10),min(1,(row+2)/6)]})
            previous_grid=grid
    finally:
        capture.release()
    windows=[]
    if rows:
        values=median_filter(np.array([x['motion_change'] for x in rows]),size=3,mode='nearest')
        centre=float(np.median(values));mad=float(np.median(np.abs(values-centre)))
        score=(values-centre)/max(.05,1.4826*mad)
        peaks,_=find_peaks(score,height=2.5,distance=max(1,round(target_fps)))
        for peak in sorted(peaks,key=lambda p:score[p],reverse=True)[:maximum_windows]:
            row=rows[peak];t=row['time']
            windows.append({'time_seconds':t,'start_seconds':max(start_seconds,t-2),
                            'end_seconds':min(end_seconds or duration,t+2),'region':row['region'],
                            'motion_outlier_score':round(float(score[peak]),3),
                            'is_collision_decision':False})
        for row,value in zip(rows,score):row['motion_outlier_score']=round(float(value),4)
    return {'method':'local_optical_flow_change','source_path':path.relative_to(ROOT).as_posix(),
            'target_fps':target_fps,'duration_seconds':duration,'scan_start_seconds':start_seconds,'scan_end_seconds':end_seconds or duration,'camera_or_cut_rejections':rejected,
            'quality_intervals':quality_intervals,
            'windows':sorted(windows,key=lambda x:x['time_seconds']),'motion_series':rows,
            'limitation':'Motion peaks occur in normal traffic too. A peak is never sufficient to declare an accident.'}
