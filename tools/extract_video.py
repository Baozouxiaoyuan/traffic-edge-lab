"""Inspect and extract continuous real-video segments without adding scene content."""
import argparse
from pathlib import Path
import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--contact',type=Path)
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--duration',type=float,default=30)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    cap=cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise ValueError('Video cannot be decoded')
    fps=cap.get(cv2.CAP_PROP_FPS)
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    print({'duration':duration,'fps':fps,'width':cap.get(3),'height':cap.get(4)})
    if args.contact:
        tiles=[]
        for time in np.linspace(args.start,min(duration-.1,args.start+args.duration),12):
            cap.set(cv2.CAP_PROP_POS_MSEC,float(time*1000));ok,frame=cap.read()
            if not ok:continue
            tile=cv2.resize(frame,(480,270))
            cv2.putText(tile,f'{time:.2f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
            tiles.append(tile)
        if tiles:
            while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(args.contact),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
    if args.output:
        if args.output.exists():raise ValueError('Output exists')
        width,height=int(cap.get(3)),int(cap.get(4))
        writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'VP80'),fps,(width,height))
        if not writer.isOpened():raise RuntimeError('Cannot encode WebM')
        cap.set(cv2.CAP_PROP_POS_MSEC,args.start*1000)
        for _ in range(round(args.duration*fps)):
            ok,frame=cap.read()
            if not ok:break
            writer.write(frame)
        writer.release()
    cap.release()


if __name__=='__main__':main()
