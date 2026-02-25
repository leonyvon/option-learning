# -*- coding: utf-8 -*-
"""
Final test of core modules
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("Option Pricing Library - Core Module Test")
print("="*60)

# Test 1: Test BSM model from models.bsm
print("\n1. Testing models.bsm...")
try:
    from option_pricing.models.bsm import bsm_model
    call_price, vega = bsm_model('call', 100, 100, 1, 0.05, 0.2)
    print(f"   bsm_model('call', 100, 100, 1, 0.05, 0.2)")
    print(f"   Result: price={call_price}, vega={vega}")
    print("   [OK] models.bsm works")
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 2: Test core.pricing
print("\n2. Testing core.pricing...")
try:
    from option_pricing.core.pricing import bsm_price
    price = bsm_price('call', 100, 100, 1, 0.05, 0.2)
    print(f"   bsm_price('call', 100, 100, 1, 0.05, 0.2) = {price}")
    print("   [OK] core.pricing works")
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 3: Test core.volatility
print("\n3. Testing core.volatility...")
try:
    from option_pricing.core.volatility import (
        calculate_implied_volatility,
        find_nearest_value,
        find_closest_number
    )
    # Test find_nearest_value
    result1 = find_nearest_value(102.5, [95, 100, 105, 110])
    result2 = find_closest_number(102.5, [95, 100, 105, 110])
    print(f"   find_nearest_value(102.5, [95, 100, 105, 110]) = {result1}")
    print(f"   find_closest_number(102.5, [95, 100, 105, 110]) = {result2}")

    # Test implied volatility
    iv = calculate_implied_volatility(10.45, 'call', 100, 100, 1, 0.05, verbose=False)
    print(f"   calculate_implied_volatility(10.45, 'call', 100, 100, 1, 0.05) = {iv:.4f}")
    print("   [OK] core.volatility works")
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test binomial model
print("\n4. Testing binomial model...")
try:
    from option_pricing.models.binomial import binary_tree_model
    price = binary_tree_model(100, 100, 1, 0.05, 0.2, 100, 'call', False)
    print(f"   binary_tree_model(100, 100, 1, 0.05, 0.2, 100, 'call', False) = {price:.4f}")
    print("   [OK] models.binomial works")
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test utils.helpers
print("\n5. Testing utils.helpers...")
try:
    from option_pricing.utils.helpers import (
        find_nearest_value as helpers_find_nearest,
        find_closest_number as helpers_find_closest
    )
    result = helpers_find_nearest(107.5, [95, 100, 105, 110])
    print(f"   helpers.find_nearest_value(107.5, ...) = {result}")
    print("   [OK] utils.helpers works")
except Exception as e:
    print(f"   [ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All core tests completed!")
print("="*60)
