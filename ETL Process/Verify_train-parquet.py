import duckdb
import os

parquet_path = "train.parquet"

# 1. Check File Size (Sanity Check)
file_size = os.path.getsize(parquet_path) / (1024**3)
print(f"File Size: {file_size:.2f} GB")

if file_size < 0.1:
    print("File is suspiciously small. Download might have failed.")

# 2. Verify Structure (Metadata Check)
try:
    con = duckdb.connect()
    # This reads the metadata footer (schema, row counts)
    print("Verifying metadata...")
    con.execute(f"DESCRIBE SELECT * FROM '{parquet_path}'")
    print("Metadata is readable.")
    
    # 3. The "Deep Read" (Forces reading the data)
    # We count rows. This forces DuckDB to scan the file structure.
    print("Scanning file to count rows (this checks for data corruption)...")
    row_count = con.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
    print(f"Data is valid. Total Rows: {row_count:,}")
    
    # 4. Preview Data
    print("\nFirst 5 rows:")
    print(con.execute(f"SELECT * FROM '{parquet_path}' LIMIT 5").df())

except Exception as e:
    print(f"\nFILE IS CORRUPT: {e}")