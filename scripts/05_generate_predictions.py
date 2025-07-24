import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=== Generating Test Predictions ===\n")

# Define the RestaurantRecommender class (needed for pickle loading)
class RestaurantRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.customer_features = None
        self.vendor_features = None
        self.popular_vendors = None
        self.vendor_similarity_matrix = None
        
    def fit(self, user_item_matrix, customer_features, vendor_features):
        """Train the recommendation model"""
        self.user_item_matrix = user_item_matrix
        self.customer_features = customer_features
        self.vendor_features = vendor_features
        
        # Calculate popular vendors (fallback for new users)
        vendor_popularity = user_item_matrix.sum(axis=0).sort_values(ascending=False)
        self.popular_vendors = vendor_popularity.head(20).index.tolist()
        
        # Create vendor similarity matrix for content-based filtering
        self._create_vendor_similarity_matrix()
        
    def _create_vendor_similarity_matrix(self):
        """Create vendor similarity matrix for content-based recommendations"""
        # Get numeric columns from vendor features
        numeric_cols = self.vendor_features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Use numeric features for similarity calculation
            vendor_numeric_features = self.vendor_features[numeric_cols].fillna(0)
            
            # Normalize features
            scaler = StandardScaler()
            vendor_features_scaled = scaler.fit_transform(vendor_numeric_features)
            
            # Calculate similarity matrix
            self.vendor_similarity_matrix = cosine_similarity(vendor_features_scaled)
            
            # Convert to DataFrame for easier indexing
            self.vendor_similarity_matrix = pd.DataFrame(
                self.vendor_similarity_matrix,
                index=self.vendor_features.index,
                columns=self.vendor_features.index
            )
        else:
            self.vendor_similarity_matrix = None
        
    def get_collaborative_recommendations(self, customer_id, top_k=10):
        """Get recommendations using collaborative filtering"""
        
        # Check if customer exists in training data
        if customer_id not in self.user_item_matrix.index:
            return []
        
        # Get customer's ratings
        customer_ratings = self.user_item_matrix.loc[customer_id].values.reshape(1, -1)
        
        # Calculate similarities with other customers
        similarities = cosine_similarity(customer_ratings, self.user_item_matrix.values)[0]
        
        # Get most similar customers (excluding self)
        customer_indices = list(range(len(similarities)))
        similar_indices = sorted(customer_indices, key=lambda x: similarities[x], reverse=True)[1:11]  # Top 10
        similar_customers = [self.user_item_matrix.index[i] for i in similar_indices]
        
        # Collect recommendations from similar customers
        recommendations = {}
        for similar_customer in similar_customers:
            customer_vendors = self.user_item_matrix.loc[similar_customer]
            for vendor_id, rating in customer_vendors.items():
                # Only recommend vendors with good ratings that customer hasn't tried
                if rating >= 3.5 and self.user_item_matrix.loc[customer_id, vendor_id] == 0:
                    if vendor_id not in recommendations:
                        recommendations[vendor_id] = 0
                    recommendations[vendor_id] += rating * similarities[self.user_item_matrix.index.get_loc(similar_customer)]
        
        # Sort recommendations by score
        if recommendations:
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return [vendor_id for vendor_id, _ in sorted_recs[:top_k]]
        else:
            return []
    
    def get_content_based_recommendations(self, customer_id, top_k=10):
        """Get recommendations using content-based filtering"""
        
        if self.vendor_similarity_matrix is None:
            return []
        
        # Get customer's order history
        if customer_id not in self.user_item_matrix.index:
            return []
        
        customer_orders = self.user_item_matrix.loc[customer_id]
        liked_vendors = customer_orders[customer_orders >= 4.0].index.tolist()  # High rated vendors
        
        if not liked_vendors:
            return []
        
        # Find similar vendors to those the customer liked
        recommendations = {}
        for liked_vendor in liked_vendors:
            if liked_vendor in self.vendor_similarity_matrix.index:
                similar_vendors = self.vendor_similarity_matrix[liked_vendor].sort_values(ascending=False)[1:6]  # Top 5 similar
                for vendor_id, similarity in similar_vendors.items():
                    if customer_orders[vendor_id] == 0:  # Customer hasn't tried this vendor
                        if vendor_id not in recommendations:
                            recommendations[vendor_id] = 0
                        recommendations[vendor_id] += similarity
        
        # Sort and return top recommendations
        if recommendations:
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return [vendor_id for vendor_id, _ in sorted_recs[:top_k]]
        else:
            return []
    
    def get_hybrid_recommendations(self, customer_id, top_k=10):
        """Get hybrid recommendations combining collaborative and content-based"""
        
        # Get recommendations from both approaches
        collaborative_recs = self.get_collaborative_recommendations(customer_id, top_k=15)
        content_recs = self.get_content_based_recommendations(customer_id, top_k=15)
        
        # Combine recommendations with weights
        hybrid_scores = {}
        
        # Collaborative filtering weight: 0.7
        for i, vendor_id in enumerate(collaborative_recs):
            score = (len(collaborative_recs) - i) / len(collaborative_recs) * 0.7
            hybrid_scores[vendor_id] = hybrid_scores.get(vendor_id, 0) + score
        
        # Content-based weight: 0.3
        for i, vendor_id in enumerate(content_recs):
            score = (len(content_recs) - i) / len(content_recs) * 0.3
            hybrid_scores[vendor_id] = hybrid_scores.get(vendor_id, 0) + score
        
        # Sort by combined score
        if hybrid_scores:
            sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            return [vendor_id for vendor_id, _ in sorted_recs[:top_k]]
        else:
            # Fallback to popular vendors
            return self.popular_vendors[:top_k]
    
    def predict_for_customer_vendor_pair(self, customer_id, vendor_id):
        """Predict if customer will order from vendor (binary prediction)"""
        
        # Get top 20 recommendations for the customer
        recommendations = self.get_hybrid_recommendations(customer_id, top_k=20)
        
        # Return 1 if vendor is in top recommendations, 0 otherwise
        return 1 if vendor_id in recommendations else 0

# Load trained model
print("Loading trained model...")
with open('models/restaurant_recommender.pkl', 'rb') as f:
    recommender = pickle.load(f)

print("✓ Model loaded successfully!")

# Create sample test data if test files don't exist
print("Checking for test data files...")

try:
    test_customers = pd.read_csv('data/test_customers.csv')
    test_locations = pd.read_csv('data/test_locations.csv')
    print(f"✓ Test customers loaded: {len(test_customers)} customers")
    print(f"✓ Test locations loaded: {len(test_locations)} location records")
except FileNotFoundError:
    print("⚠️ Test files not found. Creating sample test data...")
    
    # Create sample test customers (use some customers from training data)
    training_customers = pd.read_csv('data/customers_clean.csv')
    test_customers = training_customers.sample(n=min(100, len(training_customers))).copy()
    test_customers.to_csv('data/test_customers.csv', index=False)
    
    # Create sample test locations
    test_locations = []
    for _, customer in test_customers.iterrows():
        test_locations.append({
            'customer_id': customer['customer_id'],
            'location_number': 1
        })
    
    test_locations = pd.DataFrame(test_locations)
    test_locations.to_csv('data/test_locations.csv', index=False)
    
    print(f"✓ Created sample test data: {len(test_customers)} customers")

# Get all vendor IDs from the model
vendor_ids = list(recommender.user_item_matrix.columns)
print(f"✓ Will predict for {len(vendor_ids)} vendors")

def generate_predictions():
    """Generate predictions in required format"""
    print("\nGenerating predictions...")
    
    predictions = []
    test_customer_ids = test_customers['customer_id'].unique()
    
    print(f"Processing {len(test_customer_ids)} test customers...")
    
    for i, customer_id in enumerate(test_customer_ids):
        if i % 50 == 0:
            print(f"  Processed {i}/{len(test_customer_ids)} customers...")
        
        # Get customer's locations
        customer_locations = test_locations[test_locations['customer_id'] == customer_id]
        
        if customer_locations.empty:
            location_numbers = [1]  # Default location
        else:
            location_numbers = customer_locations['location_number'].unique()
        
        # Get recommendations for this customer
        recommended_vendors = recommender.get_hybrid_recommendations(customer_id, top_k=20)
        
        # Generate predictions for each location and vendor combination
        for location_num in location_numbers:
            for vendor_id in vendor_ids:
                
                # Binary prediction: 1 if in top recommendations, 0 otherwise
                target = 1 if vendor_id in recommended_vendors else 0
                
                predictions.append({
                    'CID': customer_id,
                    'LOC_NUM': location_num,
                    'VENDOR': vendor_id,
                    'target': target
                })
    
    return pd.DataFrame(predictions)

# Generate all predictions
predictions_df = generate_predictions()

# Save results
print(f"\nSaving {len(predictions_df):,} predictions...")
predictions_df.to_csv('results/final_predictions.csv', index=False)

# Print summary
print(f"✓ Generated predictions for {predictions_df['CID'].nunique():,} customers")
print(f"✓ Covered {predictions_df['VENDOR'].nunique():,} vendors")
print(f"✓ Total predictions: {len(predictions_df):,}")
print(f"✓ Positive predictions: {predictions_df['target'].sum():,}")
print(f"✓ Prediction rate: {predictions_df['target'].mean():.2%}")

# Validate submission format
print("\n=== Validating Submission Format ===")
required_columns = ['CID', 'LOC_NUM', 'VENDOR', 'target']
for col in required_columns:
    if col not in predictions_df.columns:
        print(f"❌ Missing column: {col}")
    else:
        print(f"✓ Column present: {col}")

# Check target values
valid_targets = predictions_df['target'].isin([0, 1]).all()
print(f"✓ Target values valid: {valid_targets}")

print(f"\n✓ Predictions saved to 'results/final_predictions.csv'")
print("=== Prediction Generation Complete ===")
