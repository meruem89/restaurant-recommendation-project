import requests
import json

# Test the API
api_url = "http://localhost:5000"

# Test 1: Check if API is running
print("=== Testing API Health ===")
response = requests.get(f"{api_url}/health")
print(f"Health Check: {response.json()}")

# Test 2: Get recommendations for a customer
print("\n=== Testing Recommendations ===")
test_data = {
    "customer_id": 12345,  # Use an actual customer ID from your data
    "location_number": 1,
    "top_k": 5
}

response = requests.post(f"{api_url}/recommend", json=test_data)
print(f"Recommendations: {json.dumps(response.json(), indent=2)}")

# Test 3: Get prediction for customer-vendor pair
print("\n=== Testing Prediction ===")
prediction_data = {
    "customer_id": 12345,
    "vendor_id": 101  # Use an actual vendor ID
}

response = requests.post(f"{api_url}/predict", json=prediction_data)
print(f"Prediction: {json.dumps(response.json(), indent=2)}")
