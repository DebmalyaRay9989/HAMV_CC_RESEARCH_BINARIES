import argparse,hashlib,os
from pathlib import Path
import pandas as pd

def sha256(path,chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--output',required=True); p.add_argument('--extensions',default='png,jpg,jpeg,tif,tiff,bmp,pdf,docx'); a=p.parse_args()
    root=Path(a.root).resolve(); exts={x.lower().lstrip('.') for x in a.extensions.split(',')}; rows=[]
    for path in sorted(x for x in root.rglob('*') if x.is_file() and x.suffix.lower().lstrip('.') in exts):
        rel=path.relative_to(root); parts=rel.parts
        source_group=parts[0] if len(parts)>1 else 'root'
        rows.append({'sample_id':rel.as_posix(),'path':str(rel),'source_group':source_group,'extension':path.suffix.lower().lstrip('.'),'size_bytes':path.stat().st_size,'sha256':sha256(path)})
    df=pd.DataFrame(rows); df.insert(0,'manifest_index',range(len(df))); Path(a.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(a.output,index=False)
    print(f'Wrote {len(df)} records to {a.output}')
if __name__=='__main__': main()
