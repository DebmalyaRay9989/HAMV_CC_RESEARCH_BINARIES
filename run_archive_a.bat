@echo off
setlocal
python run_experiments.py --config configs\real_archive.json --seeds 42,43,44,45,46,47,48,49,50,51 --ablations full,no_reliability,no_missing_mask,no_cross_view,no_cluster_consistency,fixed_k --conditions full,no_text,no_visual,no_structural,only_text,only_visual,only_structural --output runs\archive_a
python aggregate_results.py --input runs\archive_a --output results\archive_a
endlocal
