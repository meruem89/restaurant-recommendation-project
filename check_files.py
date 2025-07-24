import os
import pandas as pd

print("=== Checking Data Files ===")

# Check what files exist in data folder
data_files = os.listdir('data/')
print(f"Files in data/ folder: {data_files}")

# Look for test files
test_files = [f for f in data_files if 'test' in f.lower()]
print(f"Test files found: {test_files}")

# Check file formats
for file in test_files:
    file_path = f"data/{file}"
    size = os.path.getsize(file_path)
    print(f"{file}: {size} bytes")
    
    # Try to peek at the file
    try:
        if file.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=3)
            print(f"  ✅ CSV readable - {len(df)} rows, columns: {df.columns.tolist()}")
        elif file.endswith('.xlsx'):
            df = pd.read_excel(file_path, nrows=3, engine='openpyxl')
            print(f"  ✅ Excel readable - {len(df)} rows, columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"  ❌ Error reading {file}: {str(e)[:100]}")
