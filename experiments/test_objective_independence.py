#!/usr/bin/env python3
"""
Independence Test for Core Objective Function Module
====================================================

This test verifies that the objective function module works completely
independently of the consensus-difference metrics system.
"""

import os
import sys
import json
import tempfile
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis.objective_function import (
    evaluate_timeline_quality,
    compute_objective_function,
    load_objective_weights,
    load_top_k_keywords
)
from core.data.models import Paper


def create_test_papers() -> List[Paper]:
    """Create minimal test papers."""
    return [
        Paper(
            id="test_1",
            title="Test Paper 1",
            content="Content about machine learning and AI",
            pub_year=2020,
            cited_by_count=100,
            keywords=("machine learning", "artificial intelligence", "deep learning"),
            children=(),
            description="Test Paper 1"
        ),
        Paper(
            id="test_2", 
            title="Test Paper 2",
            content="Content about natural language processing",
            pub_year=2021,
            cited_by_count=50,
            keywords=("natural language processing", "NLP", "text analysis"),
            children=(),
            description="Test Paper 2"
        )
    ]


def test_no_config_file():
    """Test that module works without any configuration file."""
    print("Test 1: No Configuration File")
    print("-" * 40)
    
    # Temporarily hide config file
    config_path = "optimization_config.json"
    backup_path = "optimization_config.backup"
    
    config_exists = os.path.exists(config_path)
    if config_exists:
        os.rename(config_path, backup_path)
    
    try:
        # Test configuration loading
        weights = load_objective_weights()
        top_k = load_top_k_keywords()
        
        print(f"✅ Configuration loading: weights={weights}, top_k={top_k}")
        
        # Test objective function
        papers = create_test_papers()
        segments = [papers]  # Single segment
        
        result = evaluate_timeline_quality(segments)
        print(f"✅ Objective function evaluation: score={result.final_score:.3f}")
        
        print("✅ SUCCESS: Module works without configuration file")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
        
    finally:
        # Restore config file
        if config_exists:
            os.rename(backup_path, config_path)
    
    return True


def test_minimal_config():
    """Test with minimal independent configuration."""
    print("\nTest 2: Minimal Independent Configuration")
    print("-" * 40)
    
    # Create minimal config
    minimal_config = {
        "objective_function": {
            "cohesion_weight": 0.7,
            "separation_weight": 0.3,
            "top_k_keywords": 10
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(minimal_config, f, indent=2)
        temp_config_path = f.name
    
    # Temporarily replace config
    config_path = "optimization_config.json"
    backup_path = "optimization_config.backup"
    
    config_exists = os.path.exists(config_path)
    if config_exists:
        os.rename(config_path, backup_path)
    
    os.rename(temp_config_path, config_path)
    
    try:
        # Test with minimal config
        weights = load_objective_weights()
        top_k = load_top_k_keywords()
        
        expected_weights = (0.7, 0.3)
        expected_top_k = 10
        
        if weights == expected_weights and top_k == expected_top_k:
            print(f"✅ Configuration loading: weights={weights}, top_k={top_k}")
        else:
            print(f"❌ Configuration mismatch: got weights={weights}, top_k={top_k}")
            return False
        
        # Test objective function with custom weights
        papers = create_test_papers()
        segments = [papers[:1], papers[1:]]  # Two segments
        
        result = evaluate_timeline_quality(segments)
        print(f"✅ Objective function evaluation: score={result.final_score:.3f}")
        print(f"   Methodology: {result.methodology}")
        
        print("✅ SUCCESS: Module works with minimal independent configuration")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
        
    finally:
        # Restore original config
        os.remove(config_path)
        if config_exists:
            os.rename(backup_path, config_path)
    
    return True


def test_no_consensus_diff_dependency():
    """Test that module doesn't import anything from consensus_difference_metrics."""
    print("\nTest 3: No Consensus-Difference Dependencies")
    print("-" * 40)
    
    try:
        # Import objective function module
        import core.analysis.objective_function as obj_func
        
        # Check that it doesn't import consensus_difference_metrics
        module_dict = obj_func.__dict__
        
        # Look for any references to consensus_difference_metrics
        consensus_refs = [
            name for name in module_dict.keys() 
            if 'consensus' in name.lower() or 'difference' in name.lower()
        ]
        
        if consensus_refs:
            print(f"❌ Found consensus/difference references: {consensus_refs}")
            return False
        
        # Check imports
        import inspect
        source = inspect.getsource(obj_func)
        
        if 'consensus_difference_metrics' in source:
            print("❌ Found import of consensus_difference_metrics")
            return False
        
        print("✅ No imports from consensus_difference_metrics")
        print("✅ No consensus/difference references in module")
        print("✅ SUCCESS: Module is completely independent")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    return True


def test_complete_functionality():
    """Test complete functionality without any legacy dependencies."""
    print("\nTest 4: Complete Functionality Test")
    print("-" * 40)
    
    try:
        papers = create_test_papers()
        
        # Test single segment
        single_segment = [papers]
        result1 = evaluate_timeline_quality(single_segment)
        print(f"✅ Single segment: score={result1.final_score:.3f}")
        
        # Test multiple segments
        multi_segments = [[papers[0]], [papers[1]]]
        result2 = evaluate_timeline_quality(multi_segments)
        print(f"✅ Multi segments: score={result2.final_score:.3f}")
        
        # Test custom weights
        result3 = compute_objective_function(
            multi_segments,
            cohesion_weight=0.6,
            separation_weight=0.4
        )
        print(f"✅ Custom weights: score={result3.final_score:.3f}")
        
        # Verify different weights produce different results
        if result2.final_score != result3.final_score:
            print("✅ Weight customization works correctly")
        else:
            print("❌ Weight customization not working")
            return False
        
        print("✅ SUCCESS: All functionality works independently")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    return True


def main():
    """Run all independence tests."""
    print("OBJECTIVE FUNCTION INDEPENDENCE TESTS")
    print("=" * 60)
    print("Verifying complete independence from consensus-difference metrics")
    print()
    
    tests = [
        test_no_config_file,
        test_minimal_config,
        test_no_consensus_diff_dependency,
        test_complete_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"INDEPENDENCE TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - MODULE IS COMPLETELY INDEPENDENT")
        print("✅ Ready for use after consensus-difference removal")
        return 0
    else:
        print("❌ SOME TESTS FAILED - MODULE HAS DEPENDENCIES")
        return 1


if __name__ == "__main__":
    exit(main()) 