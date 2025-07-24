import requests
import json

api_url = "http://localhost:5000"

# Test 1: Check API health
print("=== Testing API Health ===")
response = requests.get(f"{api_url}/health")
print(f"Health Check: {response.json()}")

# Test 2: Get recommendations for a customer
print("\n=== Testing Recommendations ===")
test_data = {
    "customer_id": 12345,   # Replace with actual customer ID
    "location_number": 1,
    "top_k": 5
}

response = requests.post(f"{api_url}/recommend", json=test_data)
print(f"Recommendations: {json.dumps(response.json(), indent=2)}")

# Test 3: Get prediction for a customer-vendor pair
print("\n=== Testing Prediction ===")
prediction_data = {
    "customer_id": 12345,   # Replace with actual customer ID
    "vendor_id": 101        # Replace with actual vendor ID
}

response = requests.post(f"{api_url}/predict", json=prediction_data)
print(f"Prediction: {json.dumps(response.json(), indent=2)}")
