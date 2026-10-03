@echo off
setlocal
python run_experiments.py --config configs\demo.json --seeds 0-9 --output runs\demo
python aggregate_results.py --input runs\demo --output results\demo
endlocal
