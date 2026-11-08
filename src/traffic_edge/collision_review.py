"""Experimental offline collision review using motion proposals and video VQA.

Sources: https://arxiv.org/abs/2608.08867 and
https://github.com/accidentbench/ACCIDENT . Adapted to a smaller local CPU
model; their benchmark performance is not attributed to this implementation.
Collision candidates are scene-level review intervals, not legal violations.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image

from .collision_motion import propose
from .storage import atomic_write_json
from .compute import resolve_device

ROOT=Path(__file__).resolve().parents[2]
MODEL_PATH=ROOT/'models/collision/qwen3-vl-2b-instruct'
WEIGHTS_SHA256='7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
QUESTION='''Which option best describes the vehicle interaction DURING this CCTV video?
A: The vehicles continue their ordinary movements without striking each other.
B: Two vehicles visibly strike each other and their movement changes as a result.
C: The frames are insufficient to determine whether an impact occurs.
Choose from what is visible, not source captions. Already damaged stationary vehicles alone do not show a new impact. Respond with exactly one letter: A, B or C.'''
SOURCE_MARK_MASKS=((0.,0.,.30,.08),(0.,.88,.35,1.),(.85,.92,1.,1.))


def model_available()->bool:
    return all((MODEL_PATH/name).is_file() for name in ('model.safetensors','config.json','tokenizer.json','provenance.json'))


def validated_interval(start,end,duration):
    if not all(math.isfinite(x) for x in (start,end,duration)) or not 0<=start<end<=duration+.05:
        raise ValueError('Choose a valid interval inside the source video.')
    if end-start<2:
        raise ValueError('At least two seconds of video context are required.')
    if end-start>30.001:
        raise ValueError('Experimental CPU collision review is limited to 30 seconds per request.')
    return start,min(end,duration)


def scan_windows(start,end,width=4.,stride=2.):
    if end-start<=width:return [(start,end)]
    starts=list(np.arange(start,end-width+.001,stride))
    if end-width-starts[-1]>.05:starts.append(end-width)
    return [(float(t),float(min(end,t+width))) for t in starts]


def combine_labels(full,local,quality_ok=True):
    if not quality_ok:return 'uncertain'
    if full=='B' and local=='B':return 'collision_review_candidate'
    if full=='A' and local=='A':return 'no_collision_observed'
    return 'uncertain'


def window_quality_ok(flow,start,end):
    intervals=flow.get('quality_intervals')
    # Older proposal records lack timestamps, so retain conservative abstention.
    if intervals is None:return flow['camera_or_cut_rejections']==0
    return not any(left<=end and right>=start for left,right in intervals)


def frames_at(path,start,end,region=None,mask=True):
    capture=cv2.VideoCapture(str(path));frames=[]
    try:
        if not capture.isOpened():raise ValueError('Cannot decode this source video.')
        fps=capture.get(cv2.CAP_PROP_FPS);count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for t in np.linspace(start,min(end,(count-1)/fps),16):
            index=min(count-1,round(t*fps));capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,image=capture.read()
            if not ok:raise ValueError('A requested source frame could not be decoded.')
            h,w=image.shape[:2]
            if mask:
                for l,a,r,b in SOURCE_MARK_MASKS:image[int(a*h):int(b*h),int(l*w):int(r*w)]=128
            if region is not None:
                image=image[int(region[1]*h):int(region[3]*h),int(region[0]*w):int(region[2]*w)]
                if not image.size:raise ValueError('Invalid search region.')
            image=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));image.thumbnail((768,768),Image.Resampling.LANCZOS)
            frames.append((round(index/fps,3),image))
        return frames
    finally:capture.release()


class VideoActionReviewer:
    def __init__(self, device="cpu"):
        import torch
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        if not model_available():raise ValueError('Local collision-review model is not prepared.')
        metadata=json.loads((MODEL_PATH/'provenance.json').read_text(encoding='utf-8'))
        record=next((x for x in metadata.get('files',[]) if x['file']=='model.safetensors'),{})
        if record.get('sha256')!=WEIGHTS_SHA256:raise ValueError('Unexpected collision model provenance.')
        torch.set_num_threads(6)
        self.device=resolve_device(device)
        self.dtype=torch.bfloat16
        if self.device.startswith('cuda') and not torch.cuda.is_bf16_supported():
            self.dtype=torch.float16
        self.processor=AutoProcessor.from_pretrained(str(MODEL_PATH),local_files_only=True)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(str(MODEL_PATH),local_files_only=True,
                  dtype=self.dtype,attn_implementation='sdpa').to(self.device).eval()

    def classify(self,frames,fps,duration):
        import torch
        message=[{'role':'user','content':[{'type':'video'},{'type':'text','text':QUESTION}]}]
        text=self.processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
        metadata={'fps':fps,'duration':duration,'total_num_frames':round(fps*duration),'frames_indices':[round(t*fps) for t,_ in frames]}
        inputs=self.processor(text=[text],videos=[np.stack([np.asarray(image) for _,image in frames])],
            video_metadata=[metadata],do_sample_frames=False,
            size={'shortest_edge':len(frames)*128*128,'longest_edge':len(frames)*384*384},return_tensors='pt')
        inputs={k:v.to(device=self.device,dtype=self.dtype) if v.is_floating_point() else v.to(self.device) for k,v in inputs.items()}
        started=time.perf_counter()
        with torch.inference_mode():output=self.model.generate(**inputs,max_new_tokens=3,do_sample=False)
        raw=self.processor.batch_decode(output[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)[0]
        match=re.fullmatch(r'\s*([ABC])[.\s]*',raw)
        return {'label':match[1] if match else 'C','raw_response':raw,'processing_seconds':round(time.perf_counter()-started,3),
                'sample_times_seconds':[t for t,_ in frames]}


def collect_tracks(path,start,end,fps,device="cpu"):
    from .backends import UltralyticsByteTrackBackend
    backend=UltralyticsByteTrackBackend(str(ROOT/'yolov8n.pt'),.15,.5,[2,5,7],960,device=device)
    capture=cv2.VideoCapture(str(path));rows=[];index=int(start*fps);next_time=start
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        while True:
            ok,frame=capture.read()
            if not ok:break
            t=index/fps;index+=1
            if t>end:break
            if t+1e-6<next_time:continue
            next_time+=1/8
            rows.append({'time':t,'tracks':[{'id':x.track_id,'bbox':list(x.bbox)} for x in backend.track(frame)]})
        return rows
    finally:
        capture.release();del backend;gc.collect()


def roi_for(rows,dimensions,anchor,region):
    hints=[]
    if rows:
        frame=min(rows,key=lambda row:abs(row['time']-anchor));width,height=dimensions
        cx,cy=(region[0]+region[2])/2,(region[1]+region[3])/2;near=[]
        for track in frame['tracks']:
            box=[track['bbox'][i]/(width if i%2==0 else height) for i in range(4)]
            distance=((box[0]+box[2])/2-cx)**2+((box[1]+box[3])/2-cy)**2
            near.append((distance,box,track['id']))
        near=sorted(near,key=lambda x:x[0])[:2]
        if len(near)==2 and near[-1][0]<.16:
            region=[min(x[1][0] for x in near),min(x[1][1] for x in near),max(x[1][2] for x in near),max(x[1][3] for x in near)]
            hints=[x[2] for x in near if x[2]<1_000_000]
    return [max(0,region[0]-.12),max(0,region[1]-.12),min(1,region[2]+.12),min(1,region[3]+.12)],hints


def write_storyboard(path,start,end,output):
    frames=frames_at(path,start,end,mask=False)[::2]
    sheet=np.full((2*220,4*320,3),240,np.uint8)
    for i,(t,image) in enumerate(frames):
        rgb=np.asarray(image);h,w=rgb.shape[:2];scale=min(320/w,195/h)
        tile=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(round(w*scale),round(h*scale)))
        y,x=(i//4)*220,(i%4)*320
        sheet[y:y+tile.shape[0],x:x+tile.shape[1]]=tile
        cv2.putText(sheet,f'Source {t:.3f} s',(x+7,y+212),cv2.FONT_HERSHEY_SIMPLEX,.42,(45,45,45),1,cv2.LINE_AA)
    if not cv2.imwrite(str(output),sheet,[cv2.IMWRITE_JPEG_QUALITY,90]):raise OSError('Could not write review storyboard')


def review_video(path,output_dir,start,end,progress=None,device="cpu"):
    resolved_device=resolve_device(device)
    path=Path(path).resolve();output_dir=Path(output_dir)
    if not path.is_relative_to(ROOT/'data'):
        raise ValueError('Import the source into the project data directory first.')
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('Cannot open source video.')
        fps=cap.get(cv2.CAP_PROP_FPS);n=cap.get(cv2.CAP_PROP_FRAME_COUNT);dimensions=[int(cap.get(3)),int(cap.get(4))]
        if not math.isfinite(fps) or fps<=0 or n<=0:raise ValueError('Source timing is unavailable.')
        duration=n/fps
    finally:cap.release()
    start,end=validated_interval(float(start),float(end),duration)
    if not model_available():raise ValueError('Local collision-review model is unavailable.')
    output_dir.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    windows=scan_windows(start,end)
    if progress:progress(0,len(windows),'Finding motion and vehicle context')
    flow=propose(path,start_seconds=start,end_seconds=end)
    rows=collect_tracks(path,start,end,fps,device=device)
    reviewer=VideoActionReviewer(device=device);results=[]
    for index,(left,right) in enumerate(windows):
        hints=[w for w in flow['windows'] if left<=w['time_seconds']<=right]
        hint=max(hints,key=lambda w:w['motion_outlier_score']) if hints else None
        anchor=hint['time_seconds'] if hint else (left+right)/2
        region,ids=roi_for(rows,dimensions,anchor,hint['region'] if hint else [0,0,1,1])
        if progress:progress(index,len(windows),'Reviewing full scene and local interaction')
        full=reviewer.classify(frames_at(path,left,right),fps,duration)
        local=reviewer.classify(frames_at(path,left,right,region),fps,duration)
        quality_ok=window_quality_ok(flow,left,right)
        decision=combine_labels(full['label'],local['label'],quality_ok)
        item={'index':index,'start_seconds':round(left,3),'end_seconds':round(right,3),'decision':decision,
              'full':full,'local':local,'search_region_normalized':region,'roi_track_hints':ids,
              'actor_assignment':'not independently verified','precise_impact_time':None,'review_required':True}
        item['quality_flags']=[] if quality_ok else ['possible_camera_motion_or_scene_cut']
        if decision!='no_collision_observed':
            name=f'window-{index:03d}.jpg';write_storyboard(path,left,right,output_dir/name);item['storyboard']=name
        results.append(item)
        if progress:progress(index+1,len(windows),'Window reviewed')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    summary={'schema_version':1,'experimental':True,'source_sha256':digest,'video_path':str(path),
             'compute_device':resolved_device,
             'start_seconds':start,'end_seconds':end,'windows':results,'processing_seconds':round(time.perf_counter()-started,3),
             'candidate_window_count':sum(x['decision']=='collision_review_candidate' for x in results),
             'uncertain_window_count':sum(x['decision']=='uncertain' for x in results),
             'model':'Qwen/Qwen3-VL-2B-Instruct','model_revision':'89644892e4d85e24eaac8bacfd4f463576704203',
             'prompt_sha256':hashlib.sha256(QUESTION.encode()).hexdigest(),'source_mark_masks_normalized':SOURCE_MARK_MASKS,
             'limitations':['Offline, fixed-camera research review; not a validated real-time accident detector.',
                            'Overlapping windows may describe the same incident. Counts are review windows, not accident counts.',
                            'No collision observed is not proof that no collision occurred. No legal fault or precise impact time is inferred.']}
    atomic_write_json(output_dir/'collision_review.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--start',type=float,default=0);parser.add_argument('--end',type=float,required=True)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='cpu')
    args=parser.parse_args()
    def report(current,total,stage):print(json.dumps({'current_window':current,'total_windows':total,'stage':stage}),flush=True)
    review_video(args.input,args.output_dir,args.start,args.end,report,device=args.device)
