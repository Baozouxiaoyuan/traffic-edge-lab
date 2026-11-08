"""Trace visible paint into curves instead of extrapolating vehicle paths."""
from __future__ import annotations

import cv2
import numpy as np


def _thin(mask: np.ndarray) -> np.ndarray:
    image=np.pad((mask>0).astype(np.uint8),1)
    for _ in range(80):
        removed=0
        for phase in (0,1):
            c=image[1:-1,1:-1]
            ns=[image[:-2,1:-1],image[:-2,2:],image[1:-1,2:],image[2:,2:],
                image[2:,1:-1],image[2:,:-2],image[1:-1,:-2],image[:-2,:-2]]
            count=sum(ns)
            changes=sum(((ns[k]==0)&(ns[(k+1)%8]==1)).astype(np.uint8) for k in range(8))
            if phase==0:
                gate=(ns[0]*ns[2]*ns[4]==0)&(ns[2]*ns[4]*ns[6]==0)
            else:
                gate=(ns[0]*ns[2]*ns[6]==0)&(ns[0]*ns[4]*ns[6]==0)
            erase=(c==1)&(count>=2)&(count<=6)&(changes==1)&gate
            removed+=int(np.count_nonzero(erase))
            c[erase]=0
        if not removed:
            break
    return image[1:-1,1:-1]


def trace_markings(labels: np.ndarray, straight_segments: list[dict]) -> list[dict]:
    height,width=labels.shape
    mask=(labels==24).astype(np.uint8)
    skeleton=_thin(mask)
    padded=np.pad(skeleton,1)
    degree=np.zeros_like(skeleton)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if not (dy or dx):
                continue
            neighbour=padded[1+dy:1+dy+height,1+dx:1+dx+width].copy()
            if dy and dx:
                neighbour &= (padded[1+dy:1+dy+height,1:1+width]==0)&(padded[1:1+height,1+dx:1+dx+width]==0)
            degree+=neighbour
    junctions=cv2.dilate(((degree>2)&(skeleton>0)).astype(np.uint8),np.ones((3,3),np.uint8))
    skeleton[junctions>0]=0
    count,components,stats,_=cv2.connectedComponentsWithStats(skeleton,8)
    curves=[]
    scale=np.asarray([width,height])
    for index in range(1,count):
        if stats[index,cv2.CC_STAT_AREA]<30:
            continue
        ys,xs=np.where(components==index)
        original=set(zip(ys.tolist(),xs.tolist()))
        remaining=set(original)
        def neighbours_of(y,x):
            return [(y+dy,x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                    if (dy or dx) and (y+dy,x+dx) in original
                    and not (dy and dx and ((y+dy,x) in original or (y,x+dx) in original))]
        endpoints=[(y,x) for y,x in remaining if len(neighbours_of(y,x))==1]
        if not endpoints:
            continue
        point=min(endpoints)
        ordered=[]
        while point in remaining:
            remaining.remove(point);y,x=point;ordered.append([x,y])
            neighbours=[point for point in neighbours_of(y,x) if point in remaining]
            if not neighbours:
                break
            point=neighbours[0]
        points=np.asarray(ordered,np.float32)
        if len(points)<2:
            continue
        length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
        if length<max(45,min(width,height)*.07):
            continue
        simplified=cv2.approxPolyDP(points,2.5,False).reshape(-1,2)
        chord=simplified[-1]-simplified[0]
        chord_length=np.linalg.norm(chord)
        if chord_length<30 or chord_length/length<.65:
            continue
        transverse=False
        for segment in straight_segments:
            if segment.get('role')!='transverse_marking':
                continue
            a=np.asarray(segment['points'])*scale
            v=a[1]-a[0];unit=v/np.linalg.norm(v)
            if abs(np.dot(unit,chord/chord_length))<.9:
                continue
            distances=np.abs((simplified-a[0])@np.asarray([-unit[1],unit[0]]))
            if np.median(distances)<max(12,segment['stroke_width']*3):
                transverse=True
                break
        curves.append({'points':(simplified/scale).round(6).tolist(),'length':round(length,2),
                       'role':'transverse_marking' if transverse else 'boundary_curve',
                       'method':'learned_marking_skeleton','review_required':True})
    return sorted(curves,key=lambda item:-item['length'])


def lane_regions(labels: np.ndarray,curves: list[dict],raw_tracks: dict | None=None) -> list[dict]:
    """Recover whole visible lane interiors bounded by two supported curves.

    Regions touching several longitudinal boundaries remain unresolved rather
    than being presented as one lane. Short arrow strokes cannot bound a lane.
    """
    height,width=labels.shape;scale=np.asarray([width,height])
    radius=max(2,round(min(width,height)*.009))
    barriers=cv2.dilate(np.isin(labels,[24,23,2]).astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8))
    road=((labels==13)&(barriers==0)).astype(np.uint8)
    count,components,stats,_=cv2.connectedComponentsWithStats(road,8)
    results=[]
    curve_masks=[]
    for curve in curves:
        canvas=np.zeros(labels.shape,np.uint8)
        cv2.polylines(canvas,[np.round(np.asarray(curve['points'])*scale).astype(np.int32)],False,1,1)
        curve_masks.append(canvas)
    for index in range(1,count):
        area=stats[index,cv2.CC_STAT_AREA]
        if area<max(400,width*height*.0015):
            continue
        mask=(components==index).astype(np.uint8)
        neighbourhood=cv2.dilate(mask,np.ones((6*radius+5,6*radius+5),np.uint8))
        supports=[]
        for k,curve in enumerate(curves):
            if curve['role']=='transverse_marking':
                continue
            contact=np.count_nonzero((curve_masks[k]>0)&(neighbourhood>0))
            if contact>max(35,curve['length']*.25):
                supports.append(k)
        if len(supports)!=2 or min(curves[k]['length'] for k in supports)<min(width,height)*.20:
            continue
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        polygon=cv2.approxPolyDP(contour,max(2,min(width,height)*.004),True).reshape(-1,2).astype(np.float32)
        if len(polygon)<3 or len(polygon)>64:
            continue
        ys,xs=np.where(mask>0)
        cloud=np.stack([xs,ys],axis=1).astype(np.float32)
        centre,eigenvectors=cv2.PCACompute(cloud,mean=None)
        along=eigenvectors[0];centre=centre[0]
        projection=(cloud-centre)@along
        axis=np.asarray([centre+along*np.percentile(projection,10),centre+along*np.percentile(projection,90)])
        vectors=[]
        for observations in (raw_tracks or {}).values():
            inside=[np.asarray(item[:2])*scale for item in observations
                    if cv2.pointPolygonTest(polygon,tuple((np.asarray(item[:2])*scale).astype(float)),False)>=0]
            if len(inside)<6:
                continue
            vector=np.median(inside[-3:],axis=0)-np.median(inside[:3],axis=0)
            if np.linalg.norm(vector)>12:
                vectors.append(vector/np.linalg.norm(vector))
        direction=[]
        if vectors and np.linalg.norm(np.mean(vectors,axis=0))>=.8:
            oriented=axis if np.dot(axis[1]-axis[0],np.mean(vectors,axis=0))>=0 else axis[::-1]
            direction=np.clip(oriented/scale,0,1).round(6).tolist()
        results.append({'polygon':(polygon/scale).round(6).tolist(),
            'axis':np.clip(axis/scale,0,1).round(6).tolist(),
            'boundary_polylines':[curves[k]['points'] for k in supports],
            'direction':direction,'direction_status':'observed_flow_requires_legal_confirmation' if direction else 'unknown',
            'track_count':len(vectors),'method':'learned_marking_bounded_road_region',
            'review_required':True,'geometry_extent':'visible_region_between_two_supported_boundaries',
            'boundary_inset_pixels':radius})
    results.sort(key=lambda lane:np.mean(np.asarray(lane['polygon'])[:,0]))
    return [dict(lane,name=f'Marked lane {index+1}') for index,lane in enumerate(results)]
