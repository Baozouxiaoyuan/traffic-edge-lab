"""Prepare the pinned official model for the separate local collision VLM trial."""
from pathlib import Path
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Qwen/Qwen3-VL-2B-Instruct'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
EXPECTED_WEIGHTS_SHA256 = '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
destination = ROOT/'models/collision/qwen3-vl-2b-instruct'
destination.mkdir(parents=True,exist_ok=True)
print('Downloading official metadata and the hash-pinned model.',flush=True)
snapshot_download(repo_id=REPO,revision=REVISION,local_dir=destination,
                  allow_patterns=['*.json','*.txt'],max_workers=4)
weights=destination/'model.safetensors'
if not weights.exists():
    # The official Qwen mirror serves identical Safetensors bytes. Identity is
    # checked against the pinned Hugging Face LFS SHA256 before the file is used.
    url='https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct/resolve/master/model.safetensors'
    temporary=destination/'model.safetensors.partial'
    size=4255140312
    prefix=temporary.stat().st_size if temporary.exists() else 0
    step=32*1024**2
    intervals=[(begin,min(size-1,begin+step-1)) for begin in range(prefix,size,step)]
    parts=destination/'download-parts';parts.mkdir(exist_ok=True)
    def download(interval):
        begin,end=interval;part=parts/f'{begin}-{end}.part'
        if part.exists() and part.stat().st_size==end-begin+1:
            return part
        for attempt in range(3):
            try:
                offset=part.stat().st_size if part.exists() else 0
                if offset>end-begin+1:
                    offset=0
                resume=begin+offset
                with requests.get(url,headers={'Range':f'bytes={resume}-{end}'},stream=True,timeout=(20,45)) as response:
                    response.raise_for_status()
                    if response.status_code!=206 or response.headers.get('content-range','').split('/')[0]!=f'bytes {resume}-{end}':
                        raise RuntimeError('Invalid response range')
                    with part.open('ab' if offset else 'wb') as stream:
                        for chunk in response.iter_content(1024**2):
                            stream.write(chunk)
                if part.stat().st_size!=end-begin+1:
                    raise RuntimeError('Incomplete part')
                return part
            except Exception:
                if attempt==2:raise
        raise RuntimeError('Download failed')
    complete=prefix
    with ThreadPoolExecutor(max_workers=16) as pool:
        for future in as_completed([pool.submit(download,interval) for interval in intervals]):
            part=future.result();complete+=part.stat().st_size
            print(f'Model: {complete/1024**2:.0f} / {size/1024**2:.0f} MB',flush=True)
    with temporary.open('ab') as stream:
        for begin,end in intervals:
            with (parts/f'{begin}-{end}.part').open('rb') as fragment:
                for chunk in iter(lambda:fragment.read(4*1024**2),b''):stream.write(chunk)
    with temporary.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=EXPECTED_WEIGHTS_SHA256:
            raise RuntimeError('Model SHA256 mismatch')
    temporary.replace(weights)
records=[]
for path in sorted(destination.iterdir()):
    if not path.is_file() or path.suffix not in ('.json','.txt','.safetensors') or path.name=='provenance.json':
        continue
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if path.name=='model.safetensors' and digest!=EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError('Model SHA256 mismatch')
    records.append({'file':path.name,'sha256':digest,'bytes':path.stat().st_size})
(destination/'provenance.json').write_text(json.dumps({'source':f'https://huggingface.co/{REPO}',
    'official_mirror':'https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct',
    'revision':REVISION,'license':'Apache-2.0','experimental':True,'files':records},indent=2)+'\n',encoding='utf-8')
print('All files ready; official model SHA256 verified.',flush=True)
