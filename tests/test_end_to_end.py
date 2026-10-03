import json, subprocess, sys
from pathlib import Path

def test_demo_run(tmp_path):
    cfg=json.load(open('configs/demo.json')); cfg['epochs']=2; cfg['output_dir']=str(tmp_path/'runs'); c=tmp_path/'cfg.json'; json.dump(cfg,open(c,'w'))
    subprocess.run([sys.executable,'run_experiments.py','--config',str(c),'--seeds','0,1','--output',str(tmp_path/'runs')],check=True)
    assert len(list((tmp_path/'runs').rglob('run.json')))==2
    subprocess.run([sys.executable,'aggregate_results.py','--input',str(tmp_path/'runs'),'--output',str(tmp_path/'results'),'--bootstrap','100'],check=True)
    assert (tmp_path/'results'/'seed_stability.csv').exists()
