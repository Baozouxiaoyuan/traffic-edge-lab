"""Conservative alignment for accumulating fixed-camera calibration evidence."""
from __future__ import annotations

import cv2
import numpy as np


def foreground_mask(shape: tuple[int,int], boxes: list | None=None, labels: np.ndarray | None=None) -> np.ndarray:
    height,width=shape
    mask=np.zeros(shape,np.uint8)
    if labels is not None:
        mask[np.isin(labels,[19,20,21,22,52,53,54,55,56,57,58,59,60,61,62])]=1
    for box in boxes or []:
        x1,y1,x2,y2=map(float,box)
        a,b=max(0,int(np.floor(x1))-2),max(0,int(np.floor(y1))-2)
        c,d=min(width,int(np.ceil(x2))+3),min(height,int(np.ceil(y2))+3)
        if c>a and d>b:
            mask[b:d,a:c]=1
    return mask


def align_static_scene(reference: np.ndarray, sample: np.ndarray, reference_occlusion: np.ndarray,
                       sample_occlusion: np.ndarray, max_displacement: float=3) -> tuple[np.ndarray | None,dict]:
    """Return reference-to-sample transform only for a stable camera view.

    Foreground vehicles are excluded from feature matching. A clearly moved
    camera is rejected, never rescued with a translation-only fallback.
    """
    if sample.shape != reference.shape:
        return None,{'accepted':False,'reason':'different_image_shape'}
    height,width=reference.shape[:2]
    grey=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
    other=cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY)
    valid_reference=(reference_occlusion==0).astype(np.uint8)*255
    valid_sample=(sample_occlusion==0).astype(np.uint8)*255
    # Video timestamps and publisher logos are not camera-calibration features.
    band=max(1,round(height*.08))
    for valid in (valid_reference,valid_sample):
        valid[:band]=0
        valid[-band:]=0
    orb=cv2.ORB_create(nfeatures=1200,edgeThreshold=15,fastThreshold=12)
    first,descriptors1=orb.detectAndCompute(grey,valid_reference)
    second,descriptors2=orb.detectAndCompute(other,valid_sample)
    matches=[]
    if descriptors1 is not None and descriptors2 is not None and len(descriptors2)>=2:
        pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors1,descriptors2,k=2)
        matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
    if len(matches)>=20:
        src=np.asarray([first[m.queryIdx].pt for m in matches],np.float32)
        dst=np.asarray([second[m.trainIdx].pt for m in matches],np.float32)
        transform,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=2000,confidence=.99)
        if transform is not None and inliers is not None:
            good=src[inliers.ravel()>0]
            spread=np.ptp(good,axis=0) if len(good) else np.zeros(2)
            coverage=float(np.prod(spread)/(width*height))
            if len(good)>=15 and coverage>=.10:
                corners=np.asarray([[0,0],[width,0],[width,height],[0,height]],np.float64)
                mapped=corners@transform[:,:2].T+transform[:,2]
                displacement=float(np.max(np.linalg.norm(mapped-corners,axis=1)))
                accepted=displacement<=max_displacement
                return (transform if accepted else None),{
                    'accepted':accepted,'method':'background_feature_alignment',
                    'inliers':len(good),'feature_coverage':round(coverage,4),
                    'max_displacement_pixels':round(displacement,3),
                    'reason':None if accepted else 'camera_view_changed',
                }
    # Low-texture views retain the older conservative translation check.
    window=cv2.createHanningWindow((width,height),cv2.CV_32F)
    shift,response=cv2.phaseCorrelate(grey.astype(np.float32),other.astype(np.float32),window)
    accepted=bool(np.isfinite(response) and response>=.3 and np.hypot(*shift)<=max_displacement)
    transform=np.asarray([[1,0,shift[0]],[0,1,shift[1]]],np.float64) if accepted else None
    return transform,{'accepted':accepted,'method':'phase_translation_fallback',
                      'response':round(float(response),4) if np.isfinite(response) else None,
                      'reason':None if accepted else 'alignment_not_established'}
