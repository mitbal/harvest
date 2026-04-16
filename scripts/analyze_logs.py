import os
import glob
import re
from collections import Counter
import argparse

def analyze_logs(log_dir):
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    log_files = glob.glob(os.path.join(log_dir, "*.txt"))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    print(f"Found {len(log_files)} log files. Analyzing...\n")
    
    total_visits = 0
    unique_visitors = set()
    page_views = Counter()
    sources = Counter()
    redis_times = []
    stocks_searched = Counter()
    
    # Track errors and warnings
    warnings = Counter()
    
    # Regex patterns
    visit_pattern = re.compile(r"VISIT \| visitor=(.*?) \| page=(.*?) \| source=(.*?) \|")
    redis_pattern = re.compile(r"redis get .* took ([\d\.]+) seconds")
    redis_key_pattern = re.compile(r"get redis key:.*total time: ([\d\.]+) seconds")
    download_pattern = re.compile(r"Total download time for (.*?):")
    
    for filepath in log_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if "VISIT |" in line:
                        match = visit_pattern.search(line)
                        if match:
                            visitor_id = match.group(1).strip()
                            page = match.group(2).strip()
                            source = match.group(3).strip()
                            
                            total_visits += 1
                            unique_visitors.add(visitor_id)
                            page_views[page] += 1
                            if source and source != "None":
                                sources[source] += 1
                    
                    elif "redis" in line and ("took" in line or "total time:" in line):
                        t_match = redis_pattern.search(line) or redis_key_pattern.search(line)
                        if t_match:
                            try:
                                redis_times.append(float(t_match.group(1)))
                            except ValueError:
                                pass
                    
                    elif "use_container_width" in line and "will be removed" in line:
                        warnings["use_container_width deprecation"] += 1
                    elif "already connected! Connecting to a new session." in line:
                        warnings["Streamlit session reconnected"] += 1
                    elif "Total download time for" in line:
                        d_match = download_pattern.search(line)
                        if d_match:
                            stocks_searched[d_match.group(1).strip()] += 1

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # Output Report
    print("=" * 40)
    print("          LOG ANALYSIS REPORT")
    print("=" * 40)
    
    print(f"\nTotal Visits Analysed: {total_visits}")
    print(f"Total Unique Visitors: {len(unique_visitors)}")
    
    print("\n--- Page Views ---")
    for page, count in page_views.most_common():
        print(f"  {page:<25} {count:>6}")
    
    print("\n--- Traffic Sources ---")
    if not sources:
        print("  No recognized external traffic sources.")
    for source, count in sources.most_common():
        print(f"  {source:<25} {count:>6}")
            
    print("\n--- Top Searched Stocks (Stock Picker) ---")
    if not stocks_searched:
        print("  No stock search data found.")
    for stock, count in stocks_searched.most_common(10):
        print(f"  {stock:<25} {count:>6}")

    print("\n--- Redis Performance ---")
    if redis_times:
        avg_redis = sum(redis_times) / len(redis_times)
        max_redis = max(redis_times)
        print(f"  Average fetch time:      {avg_redis:.4f}s")
        print(f"  Max fetch time:          {max_redis:.4f}s")
        print(f"  Total operations parsed: {len(redis_times)}")
    else:
        print("  No redis timing logs found.")
        
    print("\n--- Warnings & Events ---")
    if not warnings:
        print("  None")
    for warn, count in warnings.most_common():
        print(f"  {warn:<35} {count:>6}")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze site logs.")
    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs")
    parser.add_argument("--dir", type=str, default=default_dir, help="Directory containing the log files")
    args = parser.parse_args()
    
    analyze_logs(args.dir)
