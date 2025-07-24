import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import os
import random
import string
warnings.filterwarnings('ignore')

# Try to import openpyxl for Excel support
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("⚠️ openpyxl not installed. Excel export will be unavailable.")

print("="*80)
print("    ENHANCED INTERACTIVE SUBMISSION FORMAT GENERATOR")
print("    Generate output in Terminal, Excel, and CSV formats")
print("="*80)

class RestaurantRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.vendor_similarity_matrix = None
        self.popular_vendors = None

    def fit_state(self, user_item_matrix, vendor_features):
        self.user_item_matrix = user_item_matrix
        num_cols = vendor_features.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            scaled = StandardScaler().fit_transform(vendor_features[num_cols].fillna(0))
            sim = cosine_similarity(scaled)
            self.vendor_similarity_matrix = pd.DataFrame(sim,
                index=vendor_features.index, columns=vendor_features.index)
        pops = user_item_matrix.sum(axis=0).sort_values(ascending=False)
        self.popular_vendors = pops.head(20).index.tolist()

    def get_hybrid(self, cid, k=20):
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
        idxs = np.argsort(sims)[-k-1:-1]
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

# Load model and data once at startup
print("\n🤖 Loading trained model and data...")
try:
    with open('models/restaurant_recommender.pkl', 'rb') as f:
        rec = pickle.load(f)
    
    ui = pd.read_csv('data/user_item_matrix.csv', index_col=0)
    vf = pd.read_csv('data/vendor_features.csv', index_col=0)
    rec.fit_state(ui, vf)
    vendor_ids = list(ui.columns)
    
    print(f"✅ Model loaded successfully!")
    print(f"✅ Available vendors: {len(vendor_ids)}")
    
except Exception as e:
    print(f"❌ Error loading model/data: {e}")
    exit()

def get_output_format_choice():
    """Ask user for output format preferences"""
    print("\n" + "="*60)
    print("📤 SELECT OUTPUT FORMATS:")
    print("="*60)
    print("  1️⃣  Terminal display only")
    print("  2️⃣  Terminal + CSV file")
    print("  3️⃣  Terminal + Excel file")
    print("  4️⃣  Terminal + CSV + Excel files")
    print("  5️⃣  All formats (Terminal + CSV + Excel + TXT)")
    print("="*60)
    
    while True:
        choice = input("🔍 Choose output format (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("❌ Invalid choice. Please select 1-5.")

def save_in_multiple_formats(submission_lines, base_filename, data_type, format_choice):
    """Save output in multiple formats based on user choice"""
    
    os.makedirs('results', exist_ok=True)
    saved_files = []
    
    # Always save as TXT for submission
    txt_file = f'results/{base_filename}.txt'
    with open(txt_file, 'w') as f:
        for line in submission_lines:
            f.write(line + '\n')
    saved_files.append(txt_file)
    
    # Create DataFrame for CSV/Excel export
    csv_data = []
    for line in submission_lines:
        parts = line.split()
        csv_data.append({
            'CID': parts[0],
            'X1': parts[1],  # The "X" separator
            'LOC_NUM': parts[2],
            'X2': parts[3],  # The "X" separator
            'VENDOR': parts[4],
            'target': parts[5],
            'FULL_FORMAT': line  # Keep the full format string
        })
    
    df = pd.DataFrame(csv_data)
    
    # Save CSV if requested
    if format_choice in ['2', '4', '5']:
        csv_file = f'results/{base_filename}.csv'
        
        # Option 1: Save with separate columns
        df.to_csv(csv_file, index=False)
        saved_files.append(csv_file)
        
        # Option 2: Save in exact format (one column)
        exact_csv_file = f'results/{base_filename}_exact_format.csv'
        pd.DataFrame({'CID_X_LOC_NUM_X_VENDOR_target': submission_lines}).to_csv(
            exact_csv_file, index=False
        )
        saved_files.append(exact_csv_file)
    
    # Save Excel if requested and available
    if format_choice in ['3', '4', '5'] and EXCEL_AVAILABLE:
        excel_file = f'results/{base_filename}.xlsx'
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: Formatted columns
            df.to_excel(writer, sheet_name='Formatted_Data', index=False)
            
            # Sheet 2: Exact format (one column)
            pd.DataFrame({
                'CID X LOC_NUM X VENDOR target': submission_lines
            }).to_excel(writer, sheet_name='Exact_Format', index=False)
            
            # Sheet 3: Statistics
            stats_data = {
                'Metric': ['Total Predictions', 'Positive Predictions', 'Negative Predictions', 
                          'Recommendation Rate', 'Unique Customers', 'Unique Vendors'],
                'Value': [
                    len(submission_lines),
                    sum(1 for line in submission_lines if line.endswith(' 1')),
                    sum(1 for line in submission_lines if line.endswith(' 0')),
                    f"{sum(1 for line in submission_lines if line.endswith(' 1'))/len(submission_lines):.2%}",
                    len(set(line.split()[0] for line in submission_lines)),
                    len(set(line.split()[4] for line in submission_lines))
                ]
            }
            pd.DataFrame(stats_data).to_excel(writer, sheet_name='Statistics', index=False)
        
        saved_files.append(excel_file)
    
    elif format_choice in ['3', '4', '5'] and not EXCEL_AVAILABLE:
        print("⚠️ Excel format requested but openpyxl not available. Install with: pip install openpyxl")
    
    return saved_files

def display_terminal_output(submission_lines, data_type, sample_size=20):
    """Display formatted output in terminal"""
    
    total_predictions = len(submission_lines)
    positive_predictions = sum(1 for line in submission_lines if line.endswith(' 1'))
    
    print(f"\n📺 TERMINAL OUTPUT - {data_type.upper()} DATA:")
    print("="*60)
    print(f"📊 Total predictions: {total_predictions:,}")
    print(f"✅ Positive predictions: {positive_predictions:,}")
    print(f"📈 Recommendation rate: {positive_predictions/total_predictions:.2%}")
    
    print(f"\n📋 SAMPLE OUTPUT (first {sample_size} lines):")
    print("-" * 40)
    for i, line in enumerate(submission_lines[:sample_size]):
        print(f"{i+1:3d}. {line}")
    
    if len(submission_lines) > sample_size:
        print(f"     ... (and {len(submission_lines)-sample_size:,} more lines)")
    print("-" * 40)

def generate_sample_test_data():
    """Generate sample test data with realistic customer IDs"""
    train_customers = pd.read_csv('data/train_customers.csv')
    
    sample_size = min(150, len(train_customers))  # Smaller sample for demo
    test_customers = train_customers.sample(n=sample_size, random_state=42).copy()
    
    # Generate realistic customer IDs
    test_customer_ids = []
    for i in range(len(test_customers)):
        if random.random() < 0.5:
            # Format like Z59FTQD
            cid = ''.join([
                random.choice(string.ascii_uppercase),
                ''.join([str(random.randint(0, 9)) for _ in range(2)]),
                ''.join([random.choice(string.ascii_uppercase) for _ in range(4)])
            ])
        else:
            # Format like 0JP29SK
            cid = ''.join([
                str(random.randint(0, 9)),
                ''.join([random.choice(string.ascii_uppercase) for _ in range(2)]),
                ''.join([str(random.randint(0, 9)) for _ in range(2)]),
                ''.join([random.choice(string.ascii_uppercase) for _ in range(2)])
            ])
        test_customer_ids.append(cid)
    
    test_customers['customer_id'] = test_customer_ids
    
    # Create test locations
    test_locations = []
    for customer_id in test_customer_ids:
        max_locations = random.randint(1, 7)
        for loc_num in range(max_locations):
            test_locations.append({
                'customer_id': customer_id,
                'location_number': loc_num
            })
    
    test_locations = pd.DataFrame(test_locations)
    return test_customers, test_locations

def generate_format_for_training_data():
    """Generate submission format for TRAINING data with multiple output options"""
    print("\n" + "="*60)
    print("🏋️  GENERATING FORMAT FOR TRAINING DATA")
    print("="*60)
    
    # Get output format choice
    format_choice = get_output_format_choice()
    
    # Load training data
    train_customers = pd.read_csv('data/train_customers.csv')
    
    # Sample subset for demonstration
    sample_size = min(100, len(train_customers))
    sample_customers = train_customers.sample(n=sample_size, random_state=42)
    
    print(f"\n📊 Processing {len(sample_customers)} training customers...")
    
    submission_lines = []
    
    for i, (_, customer_row) in enumerate(sample_customers.iterrows()):
        if i % 25 == 0:
            print(f"   Processing customer {i+1}/{len(sample_customers)}")
        
        customer_id = customer_row['customer_id']
        location_numbers = [0, 1, 2, 3]  # Multiple locations for training
        
        # Get recommendations
        try:
            recommended_vendors = rec.get_hybrid(customer_id, k=15)
            recommended_set = set(recommended_vendors)
        except:
            recommended_set = set(rec.popular_vendors[:15])
        
        # Generate format for each location-vendor combination
        for location_num in location_numbers:
            for vendor_id in vendor_ids:
                target = 1 if vendor_id in recommended_set else 0
                line = f"{customer_id} X {location_num} X {vendor_id} {target}"
                submission_lines.append(line)
    
    # Display in terminal
    display_terminal_output(submission_lines, "TRAINING")
    
    # Save in requested formats
    saved_files = save_in_multiple_formats(
        submission_lines, "training_data_format", "training", format_choice
    )
    
    print(f"\n✅ TRAINING DATA FORMAT GENERATED!")
    print(f"📁 Files saved:")
    for file in saved_files:
        print(f"   • {file}")
    
    return submission_lines

def generate_format_for_testing_data():
    """Generate submission format for TESTING data with multiple output options"""
    print("\n" + "="*60)
    print("🧪 GENERATING FORMAT FOR TESTING DATA")
    print("="*60)
    
    # Get output format choice
    format_choice = get_output_format_choice()
    
    # Generate sample test data
    test_customers, test_locations = generate_sample_test_data()
    
    print(f"\n📊 Processing {len(test_customers)} test customers...")
    print(f"📍 Total location records: {len(test_locations)}")
    
    submission_lines = []
    test_customer_ids = test_customers['customer_id'].unique()
    
    for i, customer_id in enumerate(test_customer_ids):
        if i % 25 == 0:
            print(f"   Processing customer {i+1}/{len(test_customer_ids)}")
        
        # Get customer's locations
        customer_locs = test_locations[test_locations['customer_id'] == customer_id]
        location_numbers = sorted(customer_locs['location_number'].unique())
        
        # Get recommendations
        try:
            recommended_vendors = rec.get_hybrid(customer_id, k=15)
            recommended_set = set(recommended_vendors)
        except:
            recommended_set = set(rec.popular_vendors[:15])
        
        # Generate format for each location-vendor combination
        for location_num in location_numbers:
            for vendor_id in vendor_ids:
                target = 1 if vendor_id in recommended_set else 0
                line = f"{customer_id} X {location_num} X {vendor_id} {target}"
                submission_lines.append(line)
    
    # Display in terminal
    display_terminal_output(submission_lines, "TESTING")
    
    # Save in requested formats
    saved_files = save_in_multiple_formats(
        submission_lines, "testing_data_format", "testing", format_choice
    )
    
    print(f"\n✅ TESTING DATA FORMAT GENERATED!")
    print(f"📁 Files saved:")
    for file in saved_files:
        print(f"   • {file}")
    
    return submission_lines

def generate_comparison_format():
    """Generate and compare format for BOTH training and testing data"""
    print("\n" + "="*60)
    print("🔄 GENERATING COMPARISON FORMAT - TRAINING vs TESTING")
    print("="*60)
    
    # Get output format choice (applies to both datasets)
    format_choice = get_output_format_choice()
    
    # Generate both formats (but skip individual format choice prompts)
    print("🏋️  Step 1: Generating training data format...")
    
    # Load training data
    train_customers = pd.read_csv('data/train_customers.csv')
    sample_customers = train_customers.sample(n=min(75, len(train_customers)), random_state=42)
    
    training_lines = []
    for _, customer_row in sample_customers.iterrows():
        customer_id = customer_row['customer_id']
        try:
            recommended_vendors = rec.get_hybrid(customer_id, k=15)
            recommended_set = set(recommended_vendors)
        except:
            recommended_set = set(rec.popular_vendors[:15])
        
        for location_num in [0, 1, 2, 3]:
            for vendor_id in vendor_ids:
                target = 1 if vendor_id in recommended_set else 0
                line = f"{customer_id} X {location_num} X {vendor_id} {target}"
                training_lines.append(line)
    
    print("\n🧪 Step 2: Generating testing data format...")
    
    # Generate test data
    test_customers, test_locations = generate_sample_test_data()
    testing_lines = []
    
    for customer_id in test_customers['customer_id'].unique()[:75]:  # Match sample size
        customer_locs = test_locations[test_locations['customer_id'] == customer_id]
        location_numbers = sorted(customer_locs['location_number'].unique())
        
        try:
            recommended_vendors = rec.get_hybrid(customer_id, k=15)
            recommended_set = set(recommended_vendors)
        except:
            recommended_set = set(rec.popular_vendors[:15])
        
        for location_num in location_numbers:
            for vendor_id in vendor_ids:
                target = 1 if vendor_id in recommended_set else 0
                line = f"{customer_id} X {location_num} X {vendor_id} {target}"
                testing_lines.append(line)
    
    # Display both in terminal
    print("\n📺 TERMINAL COMPARISON OUTPUT:")
    print("="*80)
    
    print("🏋️  TRAINING DATA SAMPLE:")
    print("-" * 40)
    for i, line in enumerate(training_lines[:10]):
        print(f"{i+1:3d}. {line}")
    print(f"     ... (and {len(training_lines)-10:,} more training lines)")
    
    print("\n🧪 TESTING DATA SAMPLE:")
    print("-" * 40)
    for i, line in enumerate(testing_lines[:10]):
        print(f"{i+1:3d}. {line}")
    print(f"     ... (and {len(testing_lines)-10:,} more testing lines)")
    
    # Save comparison files
    training_files = save_in_multiple_formats(
        training_lines, "comparison_training_format", "training", format_choice
    )
    testing_files = save_in_multiple_formats(
        testing_lines, "comparison_testing_format", "testing", format_choice
    )
    
    # Create combined comparison file
    comparison_file = 'results/training_vs_testing_comparison.txt'
    with open(comparison_file, 'w') as f:
        f.write("="*80 + '\n')
        f.write("RESTAURANT RECOMMENDATION SYSTEM - TRAINING vs TESTING COMPARISON\n")
        f.write("="*80 + '\n\n')
        
        f.write("TRAINING DATA FORMAT (First 20 lines):\n")
        f.write("-" * 50 + '\n')
        for line in training_lines[:20]:
            f.write(line + '\n')
        f.write(f"... ({len(training_lines)-20:,} more training predictions)\n\n")
        
        f.write("TESTING DATA FORMAT (First 20 lines):\n")
        f.write("-" * 50 + '\n')
        for line in testing_lines[:20]:
            f.write(line + '\n')
        f.write(f"... ({len(testing_lines)-20:,} more testing predictions)\n\n")
        
        f.write("COMPARISON STATISTICS:\n")
        f.write("-" * 30 + '\n')
        f.write(f"Training predictions: {len(training_lines):,}\n")
        f.write(f"Testing predictions: {len(testing_lines):,}\n")
        
        training_positive = sum(1 for line in training_lines if line.endswith(' 1'))
        testing_positive = sum(1 for line in testing_lines if line.endswith(' 1'))
        
        f.write(f"Training positive rate: {training_positive/len(training_lines):.2%}\n")
        f.write(f"Testing positive rate: {testing_positive/len(testing_lines):.2%}\n")
    
    print(f"\n✅ COMPARISON COMPLETED!")
    print(f"📁 Training files: {len(training_files)} files saved")
    print(f"📁 Testing files: {len(testing_files)} files saved")
    print(f"📁 Comparison summary: {comparison_file}")
    
    # Statistics summary
    print(f"\n📈 COMPARISON SUMMARY:")
    print(f"   🏋️  Training: {len(training_lines):,} predictions, {training_positive:,} positive ({training_positive/len(training_lines):.2%})")
    print(f"   🧪 Testing:  {len(testing_lines):,} predictions, {testing_positive:,} positive ({testing_positive/len(testing_lines):.2%})")

def main_menu():
    """Enhanced interactive menu with output format options"""
    
    # Check Excel availability at startup
    excel_status = "✅ Available" if EXCEL_AVAILABLE else "❌ Unavailable (install openpyxl)"
    
    while True:
        print("\n" + "="*80)
        print("    ENHANCED INTERACTIVE SUBMISSION FORMAT GENERATOR")
        print("    Generate CID X LOC_NUM X VENDOR target format")
        print("="*80)
        print(f"    📊 Output Formats: Terminal ✅ | CSV ✅ | Excel {excel_status}")
        print("="*80)
        print("  1️⃣  Generate format for TRAINING DATA")
        print("  2️⃣  Generate format for TESTING DATA") 
        print("  3️⃣  Generate format for BOTH datasets (with comparison)")
        print("  4️⃣  Install Excel support (openpyxl)")
        print("  5️⃣  Exit")
        print("="*80)
        
        choice = input("\n🔍 Enter your choice (1-5): ").strip()
        
        if choice == '1':
            generate_format_for_training_data()
            input("\n⏎ Press Enter to continue...")
            
        elif choice == '2':
            generate_format_for_testing_data()
            input("\n⏎ Press Enter to continue...")
            
        elif choice == '3':
            generate_comparison_format()
            input("\n⏎ Press Enter to continue...")
            
        elif choice == '4':
            print("\n📦 To install Excel support, run:")
            print("pip install openpyxl")
            print("\nThen restart this script.")
            input("\n⏎ Press Enter to continue...")
            
        elif choice == '5':
            print("\n👋 Enhanced format generation completed!")
            break
            
        else:
            print("\n❌ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main_menu()
