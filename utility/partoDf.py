import pandas as pd
import sys

def print_parquet_as_dataframe(parquet_path):
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return
    
    print(df)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_parquet>")
        sys.exit(1)

    parquet_file = sys.argv[1]
    print_parquet_as_dataframe(parquet_file)