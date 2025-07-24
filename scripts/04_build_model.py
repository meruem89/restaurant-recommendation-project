import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=== Building Recommendation Model ===\n")

# Load all feature data
print("Loading feature data...")
customer_features = pd.read_csv('data/customer_features.csv', index_col=0)
vendor_features = pd.read_csv('data/vendor_features.csv', index_col=0)
user_item_matrix = pd.read_csv('data/user_item_matrix.csv', index_col=0)
interaction_features = pd.read_csv('data/interaction_features.csv')

print(f"✓ Customer features loaded: {customer_features.shape}")
print(f"✓ Vendor features loaded: {vendor_features.shape}")
print(f"✓ User-item matrix loaded: {user_item_matrix.shape}")
print(f"✓ Interaction features loaded: {interaction_features.shape}")

class RestaurantRecommender:
    def __init__(self):
        self.user_item_matrix = None
        self.customer_features = None
        self.vendor_features = None
        self.popular_vendors = None
        self.vendor_similarity_matrix = None
        
    def fit(self, user_item_matrix, customer_features, vendor_features):
        """Train the recommendation model"""
        print("\nTraining recommendation model...")
        
        self.user_item_matrix = user_item_matrix
        self.customer_features = customer_features
        self.vendor_features = vendor_features
        
        # Calculate popular vendors (fallback for new users)
        vendor_popularity = user_item_matrix.sum(axis=0).sort_values(ascending=False)
        self.popular_vendors = vendor_popularity.head(20).index.tolist()
        
        # Create vendor similarity matrix for content-based filtering
        self._create_vendor_similarity_matrix()
        
        print(f"✓ Model trained on {len(user_item_matrix)} customers and {len(user_item_matrix.columns)} vendors")
        print(f"✓ Popular vendors identified: {len(self.popular_vendors)}")
        
    def _create_vendor_similarity_matrix(self):
        """Create vendor similarity matrix for content-based recommendations"""
        print("Creating vendor similarity matrix...")
        
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
            print("⚠️ No numeric features found for vendor similarity")
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

# Initialize and train the model
print("\n" + "="*50)
recommender = RestaurantRecommender()
recommender.fit(user_item_matrix, customer_features, vendor_features)

# Test the model with a sample customer
print("\n=== Testing Model ===")
if len(user_item_matrix) > 0:
    test_customer = user_item_matrix.index[0]
    print(f"Testing with customer: {test_customer}")
    
    # Test different recommendation approaches
    collaborative_recs = recommender.get_collaborative_recommendations(test_customer, top_k=5)
    content_recs = recommender.get_content_based_recommendations(test_customer, top_k=5)
    hybrid_recs = recommender.get_hybrid_recommendations(test_customer, top_k=5)
    
    print(f"Collaborative recommendations: {collaborative_recs}")
    print(f"Content-based recommendations: {content_recs}")
    print(f"Hybrid recommendations: {hybrid_recs}")
    
    # Test prediction for a specific vendor
    if len(hybrid_recs) > 0:
        test_vendor = hybrid_recs[0]
        prediction = recommender.predict_for_customer_vendor_pair(test_customer, test_vendor)
        print(f"Prediction for customer {test_customer} and vendor {test_vendor}: {prediction}")

# Save the trained model
print("\nSaving trained model...")
with open('models/restaurant_recommender.pkl', 'wb') as f:
    pickle.dump(recommender, f)

print("✓ Model saved successfully to 'models/restaurant_recommender.pkl'")
print("\n=== Model Training Complete ===")
