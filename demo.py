#!/usr/bin/env python3
"""
AI MindMap Mentor Demo Script
This script demonstrates the capabilities of the mindmap generation system
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🧠 {title}")
    print("=" * 60)


def print_mindmap(mindmap: Dict[str, Any]):
    """Print a formatted mindmap"""
    print(f"\n🎯 Root Goal: {mindmap['root']}")
    print(f"📊 Total Nodes: {mindmap['total_nodes']}")
    print(f"⏱️  Estimated Time: {mindmap['estimated_total_time']}")
    print(f"🎚️  Complexity Score: {mindmap['complexity_score']:.2f}")
    print(f"🕐 Generated: {mindmap['generated_at']}")

    print("\n🌳 Mindmap Structure:")
    print_nodes(mindmap["nodes"], level=0)


def print_nodes(nodes: list, level: int = 0):
    """Recursively print nodes with proper indentation"""
    for i, node in enumerate(nodes):
        indent = "  " * level
        difficulty_emoji = {
            "Beginner": "🟢",
            "Intermediate": "🟡",
            "Advanced": "🔴",
        }.get(node["difficulty"], "⚪")

        print(f"{indent}{difficulty_emoji} {node['title']}")
        print(f"{indent}   📝 {node['description']}")
        print(f"{indent}   ⏱️  {node['time_left']}")
        print(f"{indent}   🎯 {node['difficulty']}")

        if node["resources"]:
            print(f"{indent}   📚 Resources: {len(node['resources'])} links")

        if node["children"]:
            print_nodes(node["children"], level + 1)


def test_mindmap_generation(
    description: str,
    max_depth: int = 3,
    focus_area: str = None,
    time_constraint: str = None,
):
    """Test mindmap generation with given parameters"""

    request_data = {"description": description, "max_depth": max_depth}

    if focus_area:
        request_data["focus_area"] = focus_area
    if time_constraint:
        request_data["time_constraint"] = time_constraint

    print(f"\n📤 Sending request:")
    print(f"   Description: {description}")
    print(f"   Max Depth: {max_depth}")
    if focus_area:
        print(f"   Focus Area: {focus_area}")
    if time_constraint:
        print(f"   Time Constraint: {time_constraint}")

    try:
        response = requests.post(
            f"{API_BASE}/generate_mindmap",
            json=request_data,
            timeout=60,  # Longer timeout for LLM processing
        )

        if response.status_code == 200:
            mindmap = response.json()
            print_mindmap(mindmap)
            return mindmap
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print("⏰ Request timed out - LLM processing is taking longer than expected")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def demo_learning_paths():
    """Demonstrate different learning path scenarios"""

    scenarios = [
        {
            "title": "AI/ML Learning Path",
            "description": "I want to learn artificial intelligence and machine learning from scratch",
            "max_depth": 3,
            "focus_area": "Deep learning and neural networks",
            "time_constraint": "6 months",
        },
        {
            "title": "Web Development Path",
            "description": "I want to become a full-stack web developer",
            "max_depth": 3,
            "focus_area": "Modern JavaScript frameworks",
            "time_constraint": "4 months",
        },
        {
            "title": "Data Science Path",
            "description": "I want to learn data science and analytics",
            "max_depth": 3,
            "focus_area": "Statistical analysis and visualization",
            "time_constraint": "5 months",
        },
        {
            "title": "Cybersecurity Path",
            "description": "I want to learn cybersecurity and ethical hacking",
            "max_depth": 3,
            "focus_area": "Network security and penetration testing",
            "time_constraint": "8 months",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print_header(f"Demo {i}: {scenario['title']}")

        mindmap = test_mindmap_generation(
            description=scenario["description"],
            max_depth=scenario["max_depth"],
            focus_area=scenario["focus_area"],
            time_constraint=scenario["time_constraint"],
        )

        if mindmap:
            print(f"\n✅ Successfully generated mindmap for: {scenario['title']}")
        else:
            print(f"\n❌ Failed to generate mindmap for: {scenario['title']}")

        # Add delay between requests to avoid overwhelming the service
        if i < len(scenarios):
            print("\n⏳ Waiting 5 seconds before next demo...")
            time.sleep(5)


def check_service_health():
    """Check if the service is running and healthy"""
    print_header("Service Health Check")

    try:
        # Check basic health
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Basic health check passed")
        else:
            print(f"❌ Basic health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return False

    try:
        # Check detailed API health
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API health check passed")
            print(f"   Status: {health_data['status']}")

            # Check Ollama status
            ollama_status = health_data["checks"].get("ollama", {})
            if ollama_status.get("status") == "healthy":
                print("✅ Ollama is running")
            else:
                print(f"⚠️  Ollama status: {ollama_status.get('message', 'Unknown')}")

            # Check RAG service status
            rag_status = health_data["checks"].get("rag_service", {})
            if rag_status.get("status") == "healthy":
                print("✅ RAG service is running")
            else:
                print(f"⚠️  RAG service status: {rag_status.get('message', 'Unknown')}")

        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API health check failed: {e}")
        return False

    return True


def main():
    """Main demo function"""
    print("🧠 AI MindMap Mentor - Demo Script")
    print("=" * 60)
    print("This script demonstrates the mindmap generation capabilities")
    print("Make sure the service is running on http://localhost:8000")
    print("and Ollama is available with LLaMA 3 model")

    # Check service health first
    if not check_service_health():
        print("\n❌ Service is not healthy. Please check:")
        print("   1. Is the service running? (python -m uvicorn app.main:app)")
        print("   2. Is Ollama running? (ollama run llama3)")
        print("   3. Are all dependencies installed? (pip install -r requirements.txt)")
        return

    print("\n🎯 Service is healthy! Starting demos...")

    # Run demos
    demo_learning_paths()

    print_header("Demo Complete!")
    print("🎉 All demos completed!")
    print("\n💡 Try your own learning goals:")
    print("   POST http://localhost:8000/api/v1/generate_mindmap")
    print("   with your own description and parameters")
    print("\n📖 Full API documentation: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
