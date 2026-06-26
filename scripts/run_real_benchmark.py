"""Run NeuralForge benchmark on real AGNT execution data."""
import json
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add repo to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from neuralforge.benchmark import run_full_benchmark, generate_synthetic_executions

def main():
    # Load real executions
    data_path = os.path.join(repo_root, 'data', 'agnt_executions.json')
    
    if not os.path.exists(data_path):
        print('No real execution data found. Run query_agnt_db.py first.')
        sys.exit(1)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        real_executions = json.load(f)
    
    print(f'Loaded {len(real_executions)} real executions')
    
    # Filter to executions with actual data (duration > 0 or step_count > 0)
    valid = [e for e in real_executions if e.get('duration_ms', 0) > 0 or e.get('step_count', 0) > 0]
    print(f'Valid executions (with data): {len(valid)}')
    
    if len(valid) < 5:
        print('Insufficient valid executions for benchmark.')
        return
    
    # Run benchmark on real data
    print('\n=== Running Real-Data Benchmark ===')
    report_path = os.path.join(repo_root, 'data', 'real_benchmark_report.json')
    report = run_full_benchmark(valid, data_source='real_executions', seed=42, output_path=report_path)
    
    print(f'\nBenchmark complete. Report saved to: {report_path}')
    print(f'\nOverall:')
    print(f'  Total executions: {report["total_executions"]}')
    print(f'  Data source: {report["data_source"]}')
    print(f'  Wall time: {report["total_wall_time_seconds"]}s')
    
    # DataLearner results
    dl = report['components']['data_learner']
    if dl['status'] == 'success':
        print(f'\nDataLearner:')
        print(f'  Problem type: {dl["problem_type"]}')
        print(f'  Train loss: {dl["metrics"]["train_loss"]}')
        print(f'  Val loss: {dl["metrics"]["val_loss"]}')
        print(f'  Test loss: {dl["metrics"]["test_loss"]}')
        print(f'  Overfit gap: {dl["overfit_gap"]}')
        print(f'  Split: {dl["split"]}')
    
    # Pattern Engine results
    pe = report['components']['pattern_engine']
    if pe['status'] == 'success':
        print(f'\nPattern Engine:')
        print(f'  Pattern type: {pe["pattern_type"]}')
        print(f'  Confidence: {pe["confidence"]}')
        print(f'  MAE on held-out: {pe.get("mae_on_held_out", "N/A")}')
    
    # Smart Engine results
    se = report['components']['smart_engine']
    if se['status'] == 'success':
        print(f'\nSmart Engine:')
        print(f'  Decisions on held-out: {se["decisions_on_held_out"]}')
    
    # Also run on synthetic for comparison
    print('\n=== Running Synthetic Benchmark (for comparison) ===')
    synthetic = generate_synthetic_executions(n=200, seed=42)
    syn_report = run_full_benchmark(synthetic, data_source='synthetic', seed=42)
    
    syn_dl = syn_report['components']['data_learner']
    if syn_dl['status'] == 'success':
        print(f'\nSynthetic DataLearner:')
        print(f'  Train loss: {syn_dl["metrics"]["train_loss"]}')
        print(f'  Val loss: {syn_dl["metrics"]["val_loss"]}')
        print(f'  Test loss: {syn_dl["metrics"]["test_loss"]}')
        print(f'  Overfit gap: {syn_dl["overfit_gap"]}')

if __name__ == '__main__':
    main()
