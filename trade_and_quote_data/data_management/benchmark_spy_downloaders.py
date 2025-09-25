#!/usr/bin/env python3
"""
SPY Quotes Downloader Performance Benchmark

Compares performance across different versions:
1. Original (spy_quotes_downloader.py)
2. Optimized (spy_quotes_downloader_optimized.py) 
3. Max Tier (spy_quotes_downloader_max_tier.py)
4. Enhanced (spy_quotes_downloader_enhanced.py)

Usage:
    python data_management/benchmark_spy_downloaders.py --inputDir data/ --sample 100
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime
import tempfile
import shutil

def run_downloader_benchmark(script_path, input_dir, sample_size, quote_limit, additional_args=None):
    """Run a single downloader and measure performance."""
    print(f"\n{'='*60}")
    print(f"🚀 Benchmarking: {Path(script_path).name}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return None
    
    # Prepare command
    cmd = [
        "python3", script_path,
        "--inputDir", input_dir,
        "--limit", str(quote_limit),
        "--sample", str(sample_size)
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    # Measure execution time
    start_time = time.time()
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Parse output for metrics
        output = result.stdout
        error_output = result.stderr
        
        # Extract metrics from output
        metrics = {
            'script_name': Path(script_path).name,
            'duration_seconds': duration,
            'success': result.returncode == 0,
            'stdout': output,
            'stderr': error_output,
            'return_code': result.returncode
        }
        
        # Try to extract specific metrics from output
        lines = output.split('\n')
        for line in lines:
            if 'Total rows:' in line:
                try:
                    metrics['total_rows'] = int(line.split(':')[1].strip().replace(',', ''))
                except:
                    pass
            elif 'Processing speed:' in line:
                try:
                    metrics['rows_per_second'] = float(line.split(':')[1].strip().split()[0])
                except:
                    pass
            elif 'API calls:' in line:
                try:
                    metrics['api_calls'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif 'cache hit rate' in line:
                try:
                    # Look for percentage
                    parts = line.split('cache hit rate')[1].strip()
                    if '%' in parts:
                        metrics['cache_hit_rate'] = float(parts.split('%')[0].strip()) / 100
                except:
                    pass
        
        # Print results
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"✅ Success: {metrics['success']}")
        
        if metrics.get('total_rows'):
            print(f"📊 Total rows: {metrics['total_rows']:,}")
        if metrics.get('rows_per_second'):
            print(f"🚀 Speed: {metrics['rows_per_second']:.2f} rows/sec")
        if metrics.get('api_calls'):
            print(f"🔗 API calls: {metrics['api_calls']}")
        if metrics.get('cache_hit_rate'):
            print(f"💾 Cache hit rate: {metrics['cache_hit_rate']:.1%}")
        
        if not metrics['success']:
            print(f"❌ Error output: {error_output}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏰ Timeout after {duration:.2f} seconds")
        return {
            'script_name': Path(script_path).name,
            'duration_seconds': duration,
            'success': False,
            'timeout': True,
            'error': 'Timeout expired'
        }
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"❌ Error running benchmark: {e}")
        return {
            'script_name': Path(script_path).name,
            'duration_seconds': duration,
            'success': False,
            'error': str(e)
        }


def compare_results(results):
    """Compare and analyze benchmark results."""
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful runs to compare")
        return
    
    # Create comparison table
    print(f"{'Script':<35} {'Time (s)':<10} {'Rows/sec':<12} {'API Calls':<12} {'Cache Hit':<12}")
    print("-" * 80)
    
    best_time = min(r['duration_seconds'] for r in successful_results)
    best_speed = max(r.get('rows_per_second', 0) for r in successful_results)
    
    for result in results:
        name = result['script_name'][:34]
        duration = f"{result['duration_seconds']:.2f}"
        
        if result['success']:
            speed = f"{result.get('rows_per_second', 0):.1f}" if result.get('rows_per_second') else "N/A"
            api_calls = str(result.get('api_calls', 'N/A'))
            cache_rate = f"{result.get('cache_hit_rate', 0)*100:.1f}%" if result.get('cache_hit_rate') else "N/A"
            
            # Highlight best performers
            if result['duration_seconds'] == best_time:
                duration += " ⭐"
            if result.get('rows_per_second', 0) == best_speed:
                speed += " 🚀"
                
        else:
            speed = "FAILED"
            api_calls = "FAILED"
            cache_rate = "FAILED"
        
        print(f"{name:<35} {duration:<10} {speed:<12} {api_calls:<12} {cache_rate:<12}")
    
    # Performance analysis
    print(f"\n📈 PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    if len(successful_results) > 1:
        baseline = successful_results[0]  # Assume first is original
        
        for result in successful_results[1:]:
            speedup = baseline['duration_seconds'] / result['duration_seconds']
            print(f"⚡ {result['script_name']} is {speedup:.2f}x faster than {baseline['script_name']}")
            
            if result.get('rows_per_second') and baseline.get('rows_per_second'):
                throughput_improvement = result['rows_per_second'] / baseline['rows_per_second']
                print(f"🚀 Throughput improvement: {throughput_improvement:.2f}x")
            
            if result.get('api_calls') and baseline.get('api_calls'):
                api_reduction = baseline['api_calls'] / result['api_calls']
                print(f"🔗 API call reduction: {api_reduction:.2f}x fewer calls")
            
            print()


def save_benchmark_results(results, output_file=None):
    """Save benchmark results to JSON file."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"
    
    benchmark_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'total_scripts_tested': len(results),
            'successful_runs': len([r for r in results if r['success']]),
            'failed_runs': len([r for r in results if not r['success']])
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"💾 Benchmark results saved to: {output_file}")
    return output_file


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="SPY Quotes Downloader Benchmark")
    parser.add_argument("--inputDir", "-i", required=True, help="Input directory containing parquet files")
    parser.add_argument("--sample", "-s", type=int, default=100, help="Sample size for testing (default: 100)")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Quote limit for testing (default: 5)")
    parser.add_argument("--output", "-o", help="Output file for results (optional)")
    parser.add_argument("--scripts", nargs="+", help="Specific scripts to benchmark (optional)")
    
    args = parser.parse_args()
    
    # Define scripts to benchmark
    script_dir = Path(__file__).parent
    
    default_scripts = [
        ("Enhanced (Current)", script_dir / "spy_quotes_downloader.py", ["--benchmark"])
    ]
    
    if args.scripts:
        # Filter to specific scripts
        scripts_to_run = [(name, path, extra_args) for name, path, extra_args in default_scripts 
                         if path.name in args.scripts]
    else:
        scripts_to_run = default_scripts
    
    print("🎯 SPY QUOTES DOWNLOADER PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"📁 Input Directory: {args.inputDir}")
    print(f"🔬 Sample Size: {args.sample}")
    print(f"⚡ Quote Limit: {args.limit}")
    print(f"📊 Scripts to test: {len(scripts_to_run)}")
    print("=" * 80)
    
    # Check input directory
    if not os.path.exists(args.inputDir):
        print(f"❌ Input directory not found: {args.inputDir}")
        sys.exit(1)
    
    # Run benchmarks
    results = []
    
    for name, script_path, extra_args in scripts_to_run:
        print(f"\n🔄 Running benchmark: {name}")
        result = run_downloader_benchmark(
            str(script_path), 
            args.inputDir, 
            args.sample, 
            args.limit, 
            extra_args
        )
        
        if result:
            result['friendly_name'] = name
            results.append(result)
        
        # Small delay between runs
        time.sleep(2)
    
    # Analyze and compare results
    if results:
        compare_results(results)
        output_file = save_benchmark_results(results, args.output)
        
        print(f"\n🎉 Benchmark completed successfully!")
        print(f"📊 {len(results)} scripts tested")
        print(f"💾 Results saved to: {output_file}")
    else:
        print("\n❌ No benchmark results to analyze")
        sys.exit(1)


if __name__ == "__main__":
    main()