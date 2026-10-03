import json
from pathlib import Path

def test_demo_config_matches_protocol():
    cfg = json.loads(Path('configs/demo.json').read_text())
    assert cfg['epochs'] == 30
    assert cfg['n_clusters'] == 3
