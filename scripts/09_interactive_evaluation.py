import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pickle, warnings
warnings.filterwarnings('ignore')

print("="*60)
print("    RESTAURANT RECOMMENDATION MODEL EVALUATION")
print("="*60)

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

    def get_collab(self, cid, k=10):
        if cid not in self.user_item_matrix.index: return []
        vec = self.user_item_matrix.loc[cid].values.reshape(1,-1)
        sims = cosine_similarity(vec, self.user_item_matrix.values)[0]
        idxs = np.argsort(sims)[-k-1:-1]  # exclude self
        recs={}
        for i in idxs:
            other = self.user_item_matrix.index[i]
            for v,r in self.user_item_matrix.loc[other].items():
                if r>=3.5 and self.user_item_matrix.loc[cid,v]==0:
                    recs[v]=recs.get(v,0)+r*sims[i]
        return [v for v,_ in sorted(recs.items(), key=lambda x:-x[1])[:k]]

    def get_content(self, cid, k=10):
        if self.vendor_similarity_matrix is None or cid not in self.user_item_matrix.index:
            return []
        cust = self.user_item_matrix.loc[cid]
        liked = cust[cust>=4].index
        recs={}
        for v in liked:
            if v in self.vendor_similarity_matrix:
                for vid,sim in self.vendor_similarity_matrix[v].sort_values(ascending=False)[1:k+1].items():
                    if cust.get(vid,0)==0:
                        recs[vid]=recs.get(vid,0)+sim
        return [v for v,_ in sorted(recs.items(), key=lambda x:-x[1])[:k]]

    def get_hybrid(self, cid, k=10):
        col = self.get_collab(cid,15)
        cont = self.get_content(cid,15)
        scores={}
        for i,v in enumerate(col): scores[v]=scores.get(v,0)+(15-i)/15*0.7
        for i,v in enumerate(cont): scores[v]=scores.get(v,0)+(15-i)/15*0.3
        if scores: return [v for v,_ in sorted(scores.items(), key=lambda x:-x[1])[:k]]
        return self.popular_vendors[:k]

    def predict_pair(self, cid, vid):
        return 1 if vid in self.get_hybrid(cid,20) else 0

# load model & data once
print("\nLoading model and data…")
try:
    with open('models/restaurant_recommender.pkl','rb') as f:
        rec = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# ensure rec has its state
try:
    ui = pd.read_csv('data/user_item_matrix.csv', index_col=0)
    vf = pd.read_csv('data/vendor_features.csv', index_col=0)
    rec.fit_state(ui, vf)
    vendor_ids = list(ui.columns)
    print(f"✅ Data loaded: {len(vendor_ids)} vendors")
except Exception as e:
    print(f"❌ Error loading feature data: {e}")
    exit()

def load_train():
    """Load training data"""
    print("Loading training data...")
    c = pd.read_csv('data/customers_clean.csv')
    o = pd.read_csv('data/orders_clean.csv', low_memory=False)
    loc = pd.DataFrame({'customer_id':c['customer_id'],'location_number':1})
    print(f"✅ Training data loaded: {len(c)} customers")
    return c,o,loc

def load_test():
    """Load test data with multiple engine support"""
    print("Loading test data...")
    
    # Try different engines in order
    engines_to_try = ['openpyxl', 'xlrd', 'pyxlsb']
    
    for engine in engines_to_try:
        try:
            print(f"  Trying engine: {engine}")
            tc = pd.read_excel('data/test_customers.xlsx', engine=engine)
            tl = pd.read_excel('data/test_locations.xlsx', engine=engine)
            print(f"  ✅ Success with {engine}!")
            print(f"  Test customers: {len(tc)}, Test locations: {len(tl)}")
            return tc, None, tl
            
        except Exception as e:
            print(f"  ❌ Failed with {engine}: {str(e)[:50]}...")
            continue
    
    # If Excel fails, try CSV format
    try:
        print("  Trying CSV format...")
        tc = pd.read_csv('data/test_customers.csv')
        tl = pd.read_csv('data/test_locations.csv')
        print("  ✅ Success with CSV!")
        print(f"  Test customers: {len(tc)}, Test locations: {len(tl)}")
        return tc, None, tl
        
    except Exception as e:
        print(f"  ❌ CSV also failed: {str(e)[:50]}...")
    
    print("❌ All loading methods failed!")
    return None, None, None

def eval_train(cust, orders, locs):
    """Evaluate training data performance"""
    print("\n🎯 TRAINING DATA EVALUATION")
    print("="*40)
    
    sample = cust.sample(min(200,len(cust)))['customer_id']
    y_true,y_pred = [],[]
    
    print(f"Evaluating on {len(sample)} customers...")
    
    for i, cid in enumerate(sample):
        if i % 50 == 0:
            print(f"  Processing customer {i+1}/{len(sample)}")
            
        recs = set(rec.get_hybrid(cid,10))
        actual = set(orders[orders.customer_id==cid]['vendor_id'])
        for vid in vendor_ids[:50]:
            y_pred.append(1 if vid in recs else 0)
            y_true.append(1 if vid in actual else 0)
    
    print(f"\n📊 Results:")
    print(f"Accuracy:  {accuracy_score(y_true,y_pred):.4f}")
    print(f"Precision: {precision_score(y_true,y_pred,zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true,y_pred,zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true,y_pred,zero_division=0):.4f}")
    print(f"Total predictions: {len(y_pred)}")
    print(f"Positive predictions: {sum(y_pred)}")

def eval_test(cust, _, locs):
    """Evaluate test data performance"""
    print("\n🎯 TEST DATA EVALUATION")
    print("="*40)
    
    if cust is None or locs is None:
        print("❌ No test data available")
        return
    
    preds=[]
    test_customers = cust['customer_id'].unique()
    
    print(f"Evaluating {len(test_customers)} test customers...")
    
    for i, cid in enumerate(test_customers):
        if i % 100 == 0:
            print(f"  Processing customer {i+1}/{len(test_customers)}")
            
        recs = set(rec.get_hybrid(cid,20))
        
        # Get customer locations
        customer_locs = locs[locs['customer_id'] == cid]
        if customer_locs.empty:
            loc_nums = [1]
        else:
            loc_nums = customer_locs['location_number'].unique()
        
        # Generate predictions for all vendor-location combinations
        for loc_num in loc_nums:
            for vid in vendor_ids:
                preds.append(1 if vid in recs else 0)
    
    rate = np.mean(preds)
    print(f"\n📊 Results:")
    print(f"Total predictions: {len(preds):,}")
    print(f"Positive predictions: {sum(preds):,}")
    print(f"Recommendation rate: {rate:.2%}")
    print(f"Test customers: {len(test_customers):,}")
    print(f"Vendors: {len(vendor_ids):,}")

# Interactive menu with error handling
while True:
    print("\n" + "="*50)
    print("RESTAURANT RECOMMENDATION MODEL EVALUATION")
    print("="*50)
    print(" 1️⃣  Evaluate TRAINING DATA")
    print(" 2️⃣  Evaluate TEST DATA") 
    print(" 3️⃣  Exit")
    print("="*50)
    
    choice = input("🔍 Choice (1-3): ").strip()
    
    if choice == '1':
        try:
            tc, to, tl = load_train()
            eval_train(tc, to, tl)
        except Exception as e:
            print(f"❌ Training evaluation failed: {e}")
            
    elif choice == '2':
        try:
            tc, _, tl = load_test()
            if tc is not None and tl is not None:
                eval_test(tc, _, tl)
            else:
                print("❌ Could not load test data. Please check your files.")
        except Exception as e:
            print(f"❌ Test evaluation failed: {e}")
            
    elif choice == '3':
        print("👋 Evaluation completed!")
        break
    else:
        print("❌ Invalid choice. Please select 1, 2, or 3.")
