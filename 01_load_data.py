import pandas as pd
import numpy as np
import os

print("=== Restaurant Recommendation System ===")
print("Loading data files...\n")

# Set data directory
data_dir = 'data/'

# Load all data files
def load_data():
    """Load all CSV files and return as DataFrames"""
    
    try:
        # Load training data (CSV format)
        print("Loading training data...")
        customers = pd.read_csv(data_dir + 'train_customers.csv')
        locations = pd.read_csv(data_dir + 'train_locations.csv')
        orders = pd.read_csv(data_dir + 'orders.csv')
        vendors = pd.read_csv(data_dir + 'vendors.csv')
        
        print(f"✓ Customers: {len(customers):,} rows, {len(customers.columns)} columns")
        print(f"✓ Locations: {len(locations):,} rows, {len(locations.columns)} columns") 
        print(f"✓ Orders: {len(orders):,} rows, {len(orders.columns)} columns")
        print(f"✓ Vendors: {len(vendors):,} rows, {len(vendors.columns)} columns")
        
        return customers, locations, orders, vendors
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None

# Load the data
customers, locations, orders, vendors = load_data()

# Quick data exploration
if customers is not None:
    print("\n=== Data Overview ===")
    
    # Customer data preview
    print("\nCustomer Data Columns:")
    print(customers.columns.tolist())
    print("\nFirst 3 customers:")
    print(customers.head(3))
    
    # Orders data preview  
    print("\nOrders Data Columns:")
    print(orders.columns.tolist())
    print("\nFirst 3 orders:")
    print(orders.head(3))
    
    # Vendors data preview
    print("\nVendors Data Columns:")
    print(vendors.columns.tolist())
    print("\nFirst 3 vendors:")
    print(vendors.head(3))
    
    print("\n✓ Data loading completed successfully!")
else:
    print("❌ Failed to load data. Please check file paths and formats.")
