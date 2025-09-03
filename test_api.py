#!/usr/bin/env python3
"""
Simple test script for AI MindMap Mentor API
Run this after starting the server to test the endpoints
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_api_health():
    """Test the detailed API health endpoint"""
    print("\n🔍 Testing API health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False


def test_resource_categories():
    """Test getting resource categories"""
    print("\n📚 Testing resource categories...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/resources/categories")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Resource categories failed: {e}")
        return False


def test_mindmap_generation():
    """Test mindmap generation"""
    print("\n🧠 Testing mindmap generation...")

    # Test data
    test_request = {
        "description": "I want to learn Python programming in 3 months",
        "max_depth": 3,
        "focus_area": "Web development",
        "time_constraint": "3 months",
    }

    try:
        print(f"Sending request: {json.dumps(test_request, indent=2)}")
        response = requests.post(
            f"{BASE_URL}/api/v1/generate_mindmap", json=test_request
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            mindmap = response.json()
            print("✅ Mindmap generated successfully!")
            print(f"Root: {mindmap['root']}")
            print(f"Total nodes: {mindmap['total_nodes']}")
            print(f"Estimated time: {mindmap['estimated_total_time']}")
            print(f"Complexity score: {mindmap['complexity_score']}")

            # Show first few nodes
            print("\nFirst few nodes:")
            for i, node in enumerate(mindmap["nodes"][:3]):
                print(
                    f"  {i+1}. {node['title']} ({node['time_left']}) - {node['difficulty']}"
                )

            return True
        else:
            print(f"❌ Mindmap generation failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Mindmap generation test failed: {e}")
        return False


def test_root_endpoint():
    """Test the root endpoint"""
    print("\n🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧠 AI MindMap Mentor API Test Suite")
    print("=" * 50)

    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Check", test_health),
        ("API Health", test_api_health),
        ("Resource Categories", test_resource_categories),
        ("Mindmap Generation", test_mindmap_generation),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
