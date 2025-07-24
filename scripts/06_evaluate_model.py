import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

print("=== Model Evaluation ===\n")

# Load your predictions
predictions = pd.read_csv('results/final_predictions.csv')

# Calculate basic metrics
total_predictions = len(predictions)
positive_predictions = predictions['target'].sum()
prediction_rate = positive_predictions / total_predictions

print(f"Total predictions: {total_predictions:,}")
print(f"Positive predictions: {positive_predictions:,}")
print(f"Recommendation rate: {prediction_rate:.2%}")

# Analyze recommendation distribution
customer_rec_stats = predictions.groupby('CID')['target'].agg(['sum', 'count']).reset_index()
customer_rec_stats['rec_rate'] = customer_rec_stats['sum'] / customer_rec_stats['count']

print(f"\nCustomer Recommendation Statistics:")
print(f"Average recommendations per customer: {customer_rec_stats['sum'].mean():.1f}")
print(f"Min recommendations: {customer_rec_stats['sum'].min()}")
print(f"Max recommendations: {customer_rec_stats['sum'].max()}")

# Vendor popularity analysis
vendor_rec_stats = predictions.groupby('VENDOR')['target'].sum().sort_values(ascending=False)
print(f"\nTop 10 Most Recommended Vendors:")
print(vendor_rec_stats.head(10))

print("\n✓ Model evaluation complete!")
