# Run the full polynomial-autonomous WSINDy benchmark and save results to data/.
# Usage:  python run_full_benchmark.py [noise]
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarking import run_benchmark

if __name__ == '__main__':
    noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    tag = f'noise{noise:g}'.replace('.', 'p')
    prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                          'data', f'benchmark_{tag}')
    sys_df, eq_df = run_benchmark(noise=noise, csv_prefix=prefix, verbose=True)
    print(f'\nWrote {prefix}_systems.csv and {prefix}_equations.csv')
