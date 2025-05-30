"""
Test Runner for AI Agent Application
Runs unit tests, integration tests, and API tests
"""

import sys
import os
import argparse
import unittest
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def discover_and_run_tests(test_dir="tests", pattern="test_*.py", verbosity=2):
    """Discover and run tests from specified directory"""
    
    print(f"🔍 Discovering tests in {test_dir}/ with pattern {pattern}")
    
    # Discover tests
    loader = unittest.TestLoader()
    start_dir = test_dir
    suite = loader.discover(start_dir, pattern=pattern)
    
    # Count tests
    def count_tests(test_suite):
        count = 0
        for test in test_suite:
            if isinstance(test, unittest.TestSuite):
                count += count_tests(test)
            else:
                count += 1
        return count
    
    test_count = count_tests(suite)
    print(f"📊 Found {test_count} tests")
    
    if test_count == 0:
        print("⚠️ No tests found!")
        return False
    
    # Run tests
    print("🧪 Running tests...")
    print("=" * 60)
    
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("=" * 60)
    print(f"⏱️ Tests completed in {end_time - start_time:.2f} seconds")
    print(f"🧪 Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️ Errors: {len(result.errors)}")
    print(f"⏭️ Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


def run_unit_tests(verbosity=2):
    """Run unit tests only"""
    print("\n🧪 Running Unit Tests")
    print("=" * 60)
    return discover_and_run_tests("tests", "test_*.py", verbosity)


def run_api_tests(verbosity=2):
    """Run basic API tests (requires running server)"""
    print("\n🌐 Basic API Tests (requires server)")
    print("=" * 60)
    print("⚠️ Skipping API tests to avoid LLM costs")
    print("💡 To run API tests manually:")
    print("   1. Start server: python main.py server")
    print("   2. Run: python tests/test_client.py --mode demo")
    print("   3. Test language: python tests/test_polish_language.py")
    return True


def run_quick_tests():
    """Run essential tests for development without LLM costs"""
    print("\n⚡ Essential Development Tests")
    print("=" * 60)
    
    # Run only core unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        # Add essential test classes
        from tests.test_agents import TestAgentInterface, TestAgentComparison
        from tests.test_agent_service import TestLogCapture
        
        suite.addTest(loader.loadTestsFromTestCase(TestAgentInterface))
        suite.addTest(loader.loadTestsFromTestCase(TestAgentComparison))
        suite.addTest(loader.loadTestsFromTestCase(TestLogCapture))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Also run basic language test
        print("\n🇵🇱 Running Polish language test...")
        try:
            from tests.test_polish_language import test_language_consistency
            test_language_consistency()
            print("✅ Language test completed")
        except Exception as e:
            print(f"❌ Language test failed: {e}")
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"❌ Could not import test modules: {e}")
        return False


def run_comprehensive_tests(verbosity=2):
    """Run all offline tests without LLM costs"""
    print("🚀 Comprehensive Offline Test Suite")
    print("=" * 60)
    
    all_passed = True
    
    # Run unit tests
    print("\n📋 Step 1: Unit Tests")
    unit_passed = run_unit_tests(verbosity)
    all_passed = all_passed and unit_passed
    
    # Run recursion limit test
    print("\n📋 Step 2: Recursion Limit Test")
    try:
        from tests.test_recursion_limit import test_recursion_limit_basic
        test_recursion_limit_basic()
        print("✅ Recursion limit test completed")
    except Exception as e:
        print(f"❌ Recursion limit test failed: {e}")
        all_passed = False
    
    # Run language test
    print("\n📋 Step 3: Language Consistency Test")
    try:
        from tests.test_polish_language import test_language_consistency
        test_language_consistency()
        print("✅ Language test completed")
    except Exception as e:
        print(f"❌ Language test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("🎉 All offline test suites passed!")
        print("💡 For full testing with LLM:")
        print("   1. Start server: python main.py server") 
        print("   2. Run API tests: python tests/test_client.py --mode demo")
    else:
        print("⚠️ Some test suites failed!")
    
    return all_passed


def main():
    """Main test runner function"""
    
    parser = argparse.ArgumentParser(
        description="Test Runner for AI Agent Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                          # Run all unit tests
  python run_tests.py --mode unit              # Run unit tests only
  python run_tests.py --mode api               # Run API tests only  
  python run_tests.py --mode quick             # Run quick development tests
  python run_tests.py --mode comprehensive     # Run all tests
  python run_tests.py --verbosity 1            # Run with minimal output
  python run_tests.py --pattern "test_agent*"  # Run specific test pattern
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["unit", "api", "quick", "comprehensive"],
        default="unit",
        help="Test mode to run (default: unit)"
    )
    
    parser.add_argument(
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Test verbosity level (0=quiet, 1=normal, 2=verbose)"
    )
    
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="Test file pattern to match (default: test_*.py)"
    )
    
    args = parser.parse_args()
    
    print("🧪 AI Agent Application Test Runner")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Verbosity: {args.verbosity}")
    print(f"Pattern: {args.pattern}")
    
    success = False
    
    try:
        if args.mode == "unit":
            success = run_unit_tests(args.verbosity)
        elif args.mode == "api":
            success = run_api_tests(args.verbosity)
        elif args.mode == "quick":
            success = run_quick_tests()
        elif args.mode == "comprehensive":
            success = run_comprehensive_tests(args.verbosity)
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        success = False
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 