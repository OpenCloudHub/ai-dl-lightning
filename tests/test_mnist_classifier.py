#!/usr/bin/env python3
"""Test script for Fashion MNIST Classifier Ray Serve deployment"""

import numpy as np
import requests
from torchvision.datasets import FashionMNIST

BASE_URL = "http://localhost:8000"

# Fashion MNIST class names
FASHION_MNIST_CLASSES = FashionMNIST.classes


def test_health_check():
    """Test the root health check endpoint"""
    print("🔍 Testing root health check...")
    response = requests.get(f"{BASE_URL}/")

    if response.status_code == 200:
        print("✅ Health check passed")
        data = response.json()
        print(f"   Status: {data['status']}")
        print(f"   Model: {data.get('model_info', 'N/A')}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def test_prediction_with_dataset():
    """Test predictions using real Fashion MNIST images"""
    print("👕 Testing predictions with Fashion MNIST samples...")

    # Load test dataset (no transform, raw PIL images)
    dataset = FashionMNIST(root="./data", train=False, download=True, transform=None)

    # Test with 5 random samples
    test_indices = [0, 100, 500, 1000, 2000]
    images = []
    expected_labels = []

    for idx in test_indices:
        img, label = dataset[idx]
        # Convert PIL image to flattened array (784 values)
        img_array = np.array(img, dtype=np.uint8).flatten().tolist()
        images.append(img_array)
        expected_labels.append(label)

    payload = {"images": images}

    response = requests.post(
        f"{BASE_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        result = response.json()
        print("✅ Predictions successful")
        predictions = result["predictions"]

        correct = 0
        for i, (pred_obj, expected) in enumerate(zip(predictions, expected_labels)):
            pred = pred_obj["class_id"]  # Extract class_id from dict
            match = "✓" if pred == expected else "✗"
            print(
                f"   {match} Sample {test_indices[i]}: Predicted '{pred_obj['class_name']}', Expected '{FASHION_MNIST_CLASSES[expected]}' (confidence: {pred_obj['confidence']:.2f})"
            )
            if pred == expected:
                correct += 1

        accuracy = (correct / len(predictions)) * 100
        print(f"\n   Accuracy: {accuracy:.1f}% ({correct}/{len(predictions)})")
        print(f"   Timestamp: {result.get('timestamp', 'N/A')}")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def test_single_prediction():
    """Test prediction with a single Fashion MNIST image"""
    print("👟 Testing single sneaker prediction...")

    dataset = FashionMNIST(root="./data", train=False, download=True, transform=None)

    # Get a sneaker (class 7)
    sneaker_idx = None
    for i in range(len(dataset)):
        if dataset[i][1] == 7:  # Sneaker class
            sneaker_idx = i
            break

    if sneaker_idx is None:
        print("⚠️  Could not find sneaker sample")
        return

    img, label = dataset[sneaker_idx]
    img_array = np.array(img, dtype=np.uint8).flatten().tolist()

    payload = {"images": [img_array]}

    response = requests.post(f"{BASE_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        pred_obj = result["predictions"][0]  # Get first prediction object
        pred = pred_obj["class_id"]  # Extract class_id
        print("✅ Single prediction successful")
        print(
            f"   Predicted: {pred_obj['class_name']} (confidence: {pred_obj['confidence']:.2f})"
        )
        print(f"   Expected: {FASHION_MNIST_CLASSES[label]}")
        print(f"   Match: {'✓ Yes' if pred == label else '✗ No'}")
    else:
        print(f"❌ Single prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def test_invalid_input():
    """Test error handling with invalid input"""
    print("⚠️  Testing error handling...")

    # Invalid payload (wrong array length)
    invalid_payload = {"images": [[0] * 100]}  # Only 100 values instead of 784

    response = requests.post(f"{BASE_URL}/predict", json=invalid_payload)

    print(f"   Invalid input response: {response.status_code}")
    if response.status_code != 200:
        print("✅ Error handling works correctly")
        print(f"   Error message: {response.json().get('detail', 'N/A')}")
    else:
        print("⚠️  Expected error but got success")
    print()


def test_batch_sizes():
    """Test with different batch sizes"""
    print("📦 Testing different batch sizes...")

    dataset = FashionMNIST(root="./data", train=False, download=True, transform=None)

    for batch_size in [1, 5, 10]:
        images = []
        for i in range(batch_size):
            img, _ = dataset[i]
            img_array = np.array(img, dtype=np.uint8).flatten().tolist()
            images.append(img_array)

        payload = {"images": images}
        response = requests.post(f"{BASE_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            print(
                f"   ✅ Batch size {batch_size}: {len(result['predictions'])} predictions"
            )
        else:
            print(f"   ❌ Batch size {batch_size} failed: {response.status_code}")
    print()


def main():
    """Run all tests"""
    print("🧪 Starting Fashion MNIST Classifier API Tests")
    print("=" * 60)

    try:
        test_health_check()
        test_prediction_with_dataset()
        test_single_prediction()
        test_batch_sizes()
        test_invalid_input()

        print("🎉 All tests completed!")
        print("\n📖 API Documentation: http://localhost:8000/docs")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure Ray Serve is running:")
        print("   serve run src.serving.serve:app")
    except Exception as e:
        print(f"❌ Test error: {e}")


if __name__ == "__main__":
    main()
