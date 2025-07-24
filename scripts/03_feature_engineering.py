import pandas as pd
import numpy as np

print("=== Feature Engineering ===\n")

# Load cleaned data with proper handling of mixed types
print("Loading cleaned data...")
customers = pd.read_csv('data/customers_clean.csv')
orders = pd.read_csv('data/orders_clean.csv', low_memory=False)  # Fix for mixed types warning
vendors = pd.read_csv('data/vendors_clean.csv')
locations = pd.read_csv('data/locations_clean.csv')

def create_customer_features(customers_df, orders_df):
    """Create comprehensive customer features"""
    print("Creating customer features...")
    
    # Customer order statistics - only use numeric columns for aggregation
    numeric_agg = {}
    
    # Check which columns exist and are suitable for aggregation
    if 'vendor_id' in orders_df.columns:
        numeric_agg['vendor_id'] = ['count', 'nunique']
    
    if 'grand_total' in orders_df.columns:
        # Convert to numeric, replacing non-numeric values with NaN
        orders_df['grand_total'] = pd.to_numeric(orders_df['grand_total'], errors='coerce')
        numeric_agg['grand_total'] = ['mean', 'sum']
    
    if 'vendor_rating' in orders_df.columns:
        # Convert to numeric, replacing non-numeric values with NaN
        orders_df['vendor_rating'] = pd.to_numeric(orders_df['vendor_rating'], errors='coerce')
        numeric_agg['vendor_rating'] = 'mean'
    
    # Perform aggregation
    if numeric_agg:
        customer_stats = orders_df.groupby('customer_id').agg(numeric_agg)
        
        # Flatten column names manually
        new_columns = []
        for col in customer_stats.columns:
            if isinstance(col, tuple):
                if col[1] == 'count':
                    new_columns.append('total_orders')
                elif col[1] == 'nunique':
                    new_columns.append('unique_vendors')
                elif col[1] == 'mean' and col[0] == 'grand_total':
                    new_columns.append('avg_order_value')
                elif col[1] == 'sum':
                    new_columns.append('total_spent')
                elif col[1] == 'mean' and col[0] == 'vendor_rating':
                    new_columns.append('avg_rating_given')
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(str(col))
        
        customer_stats.columns = new_columns
        
        # Fill NaN values with 0 for numeric columns only
        customer_stats = customer_stats.fillna(0)
        
        # Calculate additional features
        if 'total_orders' in customer_stats.columns:
            customer_stats['order_frequency'] = customer_stats['total_orders']
        if 'unique_vendors' in customer_stats.columns:
            customer_stats['vendor_diversity'] = customer_stats['unique_vendors']
        
        # Create spending level categories
        if 'total_spent' in customer_stats.columns:
            customer_stats['spending_level'] = pd.cut(customer_stats['total_spent'], 
                                                    bins=3, labels=['Low', 'Medium', 'High'])
    else:
        # Create empty dataframe if no suitable columns found
        customer_stats = pd.DataFrame(index=orders_df['customer_id'].unique())
    
    # Merge with customer demographics
    customer_features = customers_df.set_index('customer_id').join(
        customer_stats, how='left'
    )
    
    # Fill NaN values appropriately for different column types
    for col in customer_features.columns:
        if customer_features[col].dtype in ['int64', 'float64']:
            customer_features[col] = customer_features[col].fillna(0)
        elif customer_features[col].dtype == 'object':
            customer_features[col] = customer_features[col].fillna('Unknown')
    
    print(f"✓ Created features for {len(customer_features)} customers")
    return customer_features

def create_vendor_features(vendors_df, orders_df):
    """Create vendor popularity features"""
    print("Creating vendor features...")
    
    # Convert numeric columns
    if 'vendor_rating' in orders_df.columns:
        orders_df['vendor_rating'] = pd.to_numeric(orders_df['vendor_rating'], errors='coerce')
    if 'grand_total' in orders_df.columns:
        orders_df['grand_total'] = pd.to_numeric(orders_df['grand_total'], errors='coerce')
    
    # Vendor statistics from orders
    vendor_stats = orders_df.groupby('vendor_id').agg({
        'customer_id': 'count',        # Total orders received
        'vendor_rating': 'mean',       # Average rating
        'grand_total': 'mean'          # Average order value
    })
    
    vendor_stats.columns = ['total_orders_received', 'avg_rating', 'avg_order_value']
    vendor_stats = vendor_stats.fillna(0)
    
    # Calculate vendor popularity metrics
    vendor_stats['popularity_score'] = (
        vendor_stats['total_orders_received'] * 0.6 + 
        vendor_stats['avg_rating'] * 0.4
    )
    
    # Merge with vendor info - try different possible ID columns
    if 'id' in vendors_df.columns:
        vendor_features = vendors_df.set_index('id').join(
            vendor_stats, how='left'
        )
    elif 'vendor_id' in vendors_df.columns:
        vendor_features = vendors_df.set_index('vendor_id').join(
            vendor_stats, how='left'
        )
    else:
        # Use the first column as ID if no standard ID column found
        vendor_features = vendors_df.set_index(vendors_df.columns[0]).join(
            vendor_stats, how='left'
        )
    
    # Fill NaN values
    for col in vendor_features.columns:
        if vendor_features[col].dtype in ['int64', 'float64']:
            vendor_features[col] = vendor_features[col].fillna(0)
        elif vendor_features[col].dtype == 'object':
            vendor_features[col] = vendor_features[col].fillna('Unknown')
    
    print(f"✓ Created features for {len(vendor_features)} vendors")
    return vendor_features

def create_user_item_matrix(orders_df):
    """Create user-item interaction matrix"""
    print("Creating user-item matrix...")
    
    # Convert vendor_rating to numeric
    orders_df['vendor_rating'] = pd.to_numeric(orders_df['vendor_rating'], errors='coerce')
    
    # Create matrix of customer-vendor interactions
    user_item_matrix = orders_df.pivot_table(
        index='customer_id',
        columns='vendor_id',
        values='vendor_rating',
        aggfunc='mean',
        fill_value=0
    )
    
    print(f"✓ Matrix created: {user_item_matrix.shape[0]} customers × {user_item_matrix.shape[1]} vendors")
    return user_item_matrix

def create_interaction_features(orders_df):
    """Create customer-vendor interaction patterns"""
    print("Creating interaction features...")
    
    # Calculate repeat order patterns
    repeat_orders = orders_df.groupby(['customer_id', 'vendor_id']).size().reset_index(name='repeat_count')
    
    # Find favorite vendors for each customer
    customer_favorites = repeat_orders.loc[repeat_orders.groupby('customer_id')['repeat_count'].idxmax()]
    customer_favorites = customer_favorites[['customer_id', 'vendor_id']].rename(columns={'vendor_id': 'favorite_vendor'})
    
    print(f"✓ Created interaction features for {len(customer_favorites)} customers")
    return customer_favorites

# Create all features
print("Starting feature engineering process...\n")

try:
    customer_features = create_customer_features(customers, orders)
    vendor_features = create_vendor_features(vendors, orders)
    user_item_matrix = create_user_item_matrix(orders)
    interaction_features = create_interaction_features(orders)

    # Save features
    print("\nSaving feature data...")
    customer_features.to_csv('data/customer_features.csv')
    vendor_features.to_csv('data/vendor_features.csv')
    user_item_matrix.to_csv('data/user_item_matrix.csv')
    interaction_features.to_csv('data/interaction_features.csv')

    print("✓ All features saved")
    print(f"✓ Customer features shape: {customer_features.shape}")
    print(f"✓ Vendor features shape: {vendor_features.shape}")
    print(f"✓ User-item matrix shape: {user_item_matrix.shape}")
    print(f"✓ Interaction features shape: {interaction_features.shape}")

    # Display sample features
    print("\n=== Sample Customer Features ===")
    print(customer_features.head(3))

    print("\n=== Sample Vendor Features ===")
    print(vendor_features.head(3))

    print("\n=== Feature Engineering Complete ===")

except Exception as e:
    print(f"❌ Error during feature engineering: {e}")
    print("This might be due to unexpected data structure. Let's check the data:")
    print(f"Orders columns: {orders.columns.tolist()}")
    print(f"Orders data types: {orders.dtypes}")
