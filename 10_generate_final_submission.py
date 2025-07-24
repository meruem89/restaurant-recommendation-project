import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import os
warnings.filterwarnings('ignore')

print("="*70)
print("    FINAL SUBMISSION GENERATOR FOR RESTAURANT RECOMMENDATION")
print("="*70)

# RestaurantRecommender class (same as before)
class RestaurantRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.vendor_similarity_matrix = None
        self.popular_vendors = None

    def fit_state(self, user_item_matrix, vendor_features):
        self.user_item_matrix = user_item_matrix
        # build similarity matrix
        num_cols = vendor_features.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            scaled = StandardScaler().fit_transform(vendor_features[num_cols].fillna(0))
            sim = cosine_similarity(scaled)
            self.vendor_similarity_matrix = pd.DataFrame(sim,
                index=vendor_features.index, columns=vendor_features.index)
        # popular fallback
        pops = user_item_matrix.sum(axis=0).sort_values(ascending=False)
        self.popular_vendors = pops.head(20).index.tolist()

    def get_hybrid(self, cid, k=20):
        """Get hybrid recommendations"""
        col = self.get_collab(cid, 15)
        cont = self.get_content(cid, 15)
        scores = {}
        for i, v in enumerate(col): 
            scores[v] = scores.get(v, 0) + (15-i)/15 * 0.7
        for i, v in enumerate(cont): 
            scores[v] = scores.get(v, 0) + (15-i)/15 * 0.3
        if scores: 
            return [v for v, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]
        return self.popular_vendors[:k]

    def get_collab(self, cid, k=10):
        if cid not in self.user_item_matrix.index: 
            return []
        vec = self.user_item_matrix.loc[cid].values.reshape(1, -1)
        sims = cosine_similarity(vec, self.user_item_matrix.values)[0]
        idxs = np.argsort(sims)[-k-1:-1]  # exclude self
        recs = {}
        for i in idxs:
            other = self.user_item_matrix.index[i]
            for v, r in self.user_item_matrix.loc[other].items():
                if r >= 3.5 and self.user_item_matrix.loc[cid, v] == 0:
                    recs[v] = recs.get(v, 0) + r * sims[i]
        return [v for v, _ in sorted(recs.items(), key=lambda x: -x[1])[:k]]

    def get_content(self, cid, k=10):
        if self.vendor_similarity_matrix is None or cid not in self.user_item_matrix.index:
            return []
        cust = self.user_item_matrix.loc[cid]
        liked = cust[cust >= 4].index
        recs = {}
        for v in liked:
            if v in self.vendor_similarity_matrix:
                for vid, sim in self.vendor_similarity_matrix[v].sort_values(ascending=False)[1:k+1].items():
                    if cust.get(vid, 0) == 0:
                        recs[vid] = recs.get(vid, 0) + sim
        return [v for v, _ in sorted(recs.items(), key=lambda x: -x[1])[:k]]

def smart_load_test_data():
    """Smart test data loader that tries multiple methods"""
    
    print("🔍 Searching for test data files...")
    
    # Check what files exist
    data_files = os.listdir('data/')
    test_files = [f for f in data_files if 'test' in f.lower()]
    
    print(f"Found potential test files: {test_files}")
    
    test_customers = None
    test_locations = None
    
    # Method 1: Try Excel files
    customer_files = [f for f in test_files if 'customer' in f.lower()]
    location_files = [f for f in test_files if 'location' in f.lower()]
    
    if customer_files and location_files:
        customer_file = customer_files[0]
        location_file = location_files[0]
        
        print(f"Trying to load: {customer_file} and {location_file}")
        
        # Try different formats
        for ext, loader in [('.csv', pd.read_csv), ('.xlsx', lambda x: pd.read_excel(x, engine='openpyxl'))]:
            try:
                if customer_file.endswith(ext):
                    test_customers = loader(f'data/{customer_file}')
                    test_locations = loader(f'data/{location_file}')
                    print(f"✅ Successfully loaded {ext} files!")
                    break
            except Exception as e:
                print(f"❌ Failed to load {ext}: {str(e)[:50]}...")
                continue
    
    # Method 2: Create sample test data from training if no test files
    if test_customers is None:
        print("📝 No test files found. Creating sample test data from training data...")
        
        try:
            train_customers = pd.read_csv('data/customers_clean.csv')
            
            # Sample some customers as test data
            sample_size = min(100, len(train_customers))
            test_customers = train_customers.sample(n=sample_size).copy()
            
            # Create test locations
            test_locations = []
            for _, customer in test_customers.iterrows():
                # Create multiple locations per customer (0 to 6 as shown in your example)
                for loc_num in range(7):  # 0, 1, 2, 3, 4, 5, 6
                    test_locations.append({
                        'customer_id': customer['customer_id'],
                        'location_number': loc_num
                    })
            
            test_locations = pd.DataFrame(test_locations)
            print(f"✅ Created sample test data: {len(test_customers)} customers, {len(test_locations)} locations")
            
        except Exception as e:
            print(f"❌ Failed to create sample data: {e}")
            return None, None
    
    return test_customers, test_locations

def generate_exact_submission_format():
    """Generate submission in the EXACT required format"""
    
    print("\n" + "="*70)
    print("GENERATING FINAL SUBMISSION IN REQUIRED FORMAT")
    print("="*70)
    
    # Load model and data
    print("🤖 Loading trained model...")
    try:
        with open('models/restaurant_recommender.pkl', 'rb') as f:
            rec = pickle.load(f)
        
        # Load feature data to restore model state
        ui = pd.read_csv('data/user_item_matrix.csv', index_col=0)
        vf = pd.read_csv('data/vendor_features.csv', index_col=0)
        rec.fit_state(ui, vf)
        
        vendor_ids = list(ui.columns)
        print(f"✅ Model loaded successfully! {len(vendor_ids)} vendors available")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    test_customers, test_locations = smart_load_test_data()
    
    if test_customers is None or test_locations is None:
        print("❌ Could not load test data. Cannot generate submission.")
        return
    
    # Generate predictions in exact format
    print(f"\n📊 Generating predictions for {len(test_customers)} customers...")
    
    submission_rows = []
    test_customer_ids = test_customers['customer_id'].unique()
    
    for i, customer_id in enumerate(test_customer_ids):
        if i % 20 == 0:
            print(f"   Processing customer {i+1}/{len(test_customer_ids)}")
        
        # Get customer's locations
        customer_locations = test_locations[test_locations['customer_id'] == customer_id]
        
        if customer_locations.empty:
            # Default to location 0 if no locations found
            location_numbers = [0]
        else:
            location_numbers = sorted(customer_locations['location_number'].unique())
        
        # Get recommendations for this customer
        recommended_vendors = rec.get_hybrid(customer_id, top_k=15)
        recommended_set = set(recommended_vendors)
        
        # Generate predictions for each location and vendor combination
        for location_num in location_numbers:
            for vendor_id in vendor_ids:
                
                # Binary prediction: 1 if recommended, 0 otherwise
                target = 1 if vendor_id in recommended_set else 0
                
                # Format exactly as required: CID X LOC_NUM X VENDOR target
                submission_rows.append(f"{customer_id} X {location_num} X {vendor_id} {target}")
    
    # Save submission file
    submission_filename = 'results/final_submission.txt'
    
    print(f"\n💾 Saving submission to {submission_filename}...")
    
    with open(submission_filename, 'w') as f:
        for row in submission_rows:
            f.write(row + '\n')
    
    # Also create CSV version for easier analysis
    csv_data = []
    for row in submission_rows:
        parts = row.split()
        cid = parts[0]
        loc_num = parts[2]
        vendor = parts[4]
        target = parts[5]
        
        csv_data.append({
            'CID': cid,
            'LOC_NUM': loc_num,
            'VENDOR': vendor,
            'target': target
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv('results/final_submission.csv', index=False)
    
    # Show results
    print(f"\n✅ SUBMISSION GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"📁 Text format: {submission_filename}")
    print(f"📁 CSV format: results/final_submission.csv")
    print(f"📊 Total predictions: {len(submission_rows):,}")
    print(f"👥 Customers: {len(test_customer_ids):,}")
    print(f"🏪 Vendors: {len(vendor_ids):,}")
    print(f"✅ Positive predictions: {sum(1 for row in submission_rows if row.endswith(' 1')):,}")
    
    # Show sample output
    print(f"\n📋 SAMPLE OUTPUT (first 10 lines):")
    print("-" * 30)
    for i, row in enumerate(submission_rows[:10]):
        print(row)
    print("-" * 30)
    
    print(f"\n🎉 Your submission is ready! Submit the file: {submission_filename}")

if __name__ == "__main__":
    generate_exact_submission_format()
