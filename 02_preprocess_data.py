import pandas as pd
import numpy as np
from datetime import datetime

print("=== Data Preprocessing ===\n")

# Load data (reuse from previous script)
data_dir = 'data/'
customers = pd.read_csv(data_dir + 'train_customers.csv')
locations = pd.read_csv(data_dir + 'train_locations.csv')
orders = pd.read_csv(data_dir + 'orders.csv')
vendors = pd.read_csv(data_dir + 'vendors.csv')

def preprocess_customers(df):
    """Clean and enhance customer data"""
    print("Processing customer data...")
    
    # Handle missing values
    df = df.copy()
    
    # Calculate age if date of birth exists
    if 'dob' in df.columns or 'birth_year' in df.columns:
        try:
            if 'dob' in df.columns:
                df['age'] = 2025 - pd.to_numeric(df['dob'], errors='coerce')
            else:
                df['age'] = 2025 - pd.to_numeric(df['birth_year'], errors='coerce')
            df['age'].fillna(df['age'].median(), inplace=True)
        except:
            df['age'] = 30  # Default age
    
    # Handle categorical variables
    if 'gender' in df.columns:
        df['gender'].fillna('Unknown', inplace=True)
    
    if 'status' in df.columns:
        df['status'].fillna('active', inplace=True)
    
    print(f"✓ Processed {len(df)} customers")
    return df

def preprocess_orders(df):
    """Clean and enhance order data"""
    print("Processing order data...")
    
    df = df.copy()
    
    # Handle missing ratings
    if 'vendor_rating' in df.columns:
        df['vendor_rating'].fillna(3.0, inplace=True)  # Neutral rating
    
    if 'driver_rating' in df.columns:
        df['driver_rating'].fillna(3.0, inplace=True)
    
    # Handle missing order values
    if 'grand_total' in df.columns:
        df['grand_total'].fillna(df['grand_total'].median(), inplace=True)
    
    # Convert timestamps if they exist
    time_columns = ['created_at', 'delivered_at', 'accepted_at']
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print(f"✓ Processed {len(df)} orders")
    return df

def preprocess_vendors(df):
    """Clean vendor data"""
    print("Processing vendor data...")
    
    df = df.copy()
    
    # Handle missing vendor tags
    if 'vendor_tag_name' in df.columns:
        df['vendor_tag_name'].fillna('restaurant', inplace=True)
    
    print(f"✓ Processed {len(df)} vendors")
    return df

def preprocess_locations(df):
    """Clean location data"""
    print("Processing location data...")
    
    df = df.copy()
    
    # Handle missing coordinates (fill with median values)
    if 'latitude' in df.columns:
        df['latitude'].fillna(df['latitude'].median(), inplace=True)
    if 'longitude' in df.columns:
        df['longitude'].fillna(df['longitude'].median(), inplace=True)
    
    print(f"✓ Processed {len(df)} locations")
    return df

# Process all data
customers_clean = preprocess_customers(customers)
orders_clean = preprocess_orders(orders)
vendors_clean = preprocess_vendors(vendors)
locations_clean = preprocess_locations(locations)

# Save cleaned data
print("\nSaving cleaned data...")
customers_clean.to_csv('data/customers_clean.csv', index=False)
orders_clean.to_csv('data/orders_clean.csv', index=False)
vendors_clean.to_csv('data/vendors_clean.csv', index=False)
locations_clean.to_csv('data/locations_clean.csv', index=False)

print("✓ All cleaned data saved to CSV files")
print("\n=== Preprocessing Complete ===")
