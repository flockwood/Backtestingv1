#!/usr/bin/env python3
"""
Setup validation script to verify the backtesting system is properly configured.
Run this after installation to ensure everything is working correctly.
"""

import sys
import os
import importlib
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(text)
    print("="*60)


def check_python_version():
    """Check Python version."""
    print("\n1. Python Version:")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.7+")
        return False


def check_required_packages():
    """Check required packages are installed."""
    print("\n2. Required Packages:")
    
    required = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'requests': 'HTTP requests'
    }
    
    all_installed = True
    for package, description in required.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✓ {package} ({version}) - {description}")
        except ImportError:
            print(f"   ✗ {package} - {description} - NOT INSTALLED")
            all_installed = False
    
    return all_installed


def check_optional_packages():
    """Check optional packages."""
    print("\n3. Optional Packages:")
    
    optional = {
        'alpaca_py': 'Alpaca Markets API',
        'scipy': 'Advanced statistics',
        'matplotlib': 'Plotting and charts',
        'openpyxl': 'Excel export',
        'psycopg2': 'PostgreSQL database',
        'pytest': 'Testing framework'
    }
    
    for package, description in optional.items():
        try:
            if package == 'alpaca_py':
                import alpaca
                version = getattr(alpaca, '__version__', 'unknown')
            elif package == 'psycopg2':
                import psycopg2
                version = psycopg2.__version__
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
            print(f"   ✓ {package} ({version}) - {description}")
        except ImportError:
            print(f"   ○ {package} - {description} - Not installed (optional)")
    
    return True


def check_project_structure():
    """Check project directory structure."""
    print("\n4. Project Structure:")
    
    required_dirs = [
        'core',
        'strategies', 
        'data',
        'analysis',
        'database',
        'tests'
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_present = True
    
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print(f"   ✓ {dir_name}/ directory")
        else:
            print(f"   ✗ {dir_name}/ directory - MISSING")
            all_present = False
    
    for file_name in required_files:
        if Path(file_name).is_file():
            print(f"   ✓ {file_name}")
        else:
            print(f"   ✗ {file_name} - MISSING")
            all_present = False
    
    return all_present


def check_configuration():
    """Check configuration files."""
    print("\n5. Configuration:")
    
    config_exists = Path('config.py').exists()
    example_exists = Path('config.py.example').exists()
    
    if config_exists:
        print("   ✓ config.py exists")
        try:
            import config
            if hasattr(config, 'ALPACA_API_KEY'):
                if config.ALPACA_API_KEY != "your_alpaca_api_key_here":
                    print("   ✓ Alpaca API key configured")
                else:
                    print("   ⚠ Alpaca API key not configured (using placeholder)")
            else:
                print("   ⚠ Alpaca API key not found in config")
        except ImportError as e:
            print(f"   ⚠ Could not import config: {e}")
    else:
        print("   ⚠ config.py not found")
        if example_exists:
            print("   ℹ Copy config.py.example to config.py and update with your credentials")
        else:
            print("   ✗ config.py.example also missing!")
    
    return example_exists


def check_data_directory():
    """Check data directory and sample data."""
    print("\n6. Data Directory:")
    
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        print("   ✓ Created data/ directory")
    else:
        print("   ✓ data/ directory exists")
    
    # Check for any CSV files
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        print(f"   ✓ Found {len(csv_files)} CSV file(s)")
        for csv_file in csv_files[:3]:  # Show first 3
            print(f"      - {csv_file.name}")
    else:
        print("   ℹ No CSV files found (will use Alpaca API)")
    
    return True


def test_imports():
    """Test importing core modules."""
    print("\n7. Module Imports:")
    
    modules_to_test = [
        ('core', 'Core engine'),
        ('strategies', 'Trading strategies'),
        ('data', 'Data loaders'),
        ('analysis', 'Performance analysis')
    ]
    
    all_imported = True
    
    for module_name, description in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"   ✓ {module_name} - {description}")
        except ImportError as e:
            print(f"   ✗ {module_name} - {description} - Error: {e}")
            all_imported = False
    
    return all_imported


def run_simple_test():
    """Run a simple functionality test."""
    print("\n8. Functionality Test:")
    
    try:
        # Try to create basic objects
        from core import BacktestEngine
        from strategies import SMAStrategy
        
        engine = BacktestEngine(initial_capital=10000)
        strategy = SMAStrategy(short_window=10, long_window=20)
        
        print("   ✓ Can create BacktestEngine")
        print("   ✓ Can create Strategy")
        
        # Test data generation
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        test_data = pd.DataFrame({
            'Open': np.random.uniform(99, 101, len(dates)),
            'High': np.random.uniform(101, 103, len(dates)),
            'Low': np.random.uniform(97, 99, len(dates)),
            'Close': np.random.uniform(98, 102, len(dates)),
            'Volume': np.random.randint(1000000, 2000000, len(dates))
        }, index=dates)
        
        signals = strategy.generate_signals(test_data)
        print("   ✓ Strategy can generate signals")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Functionality test failed: {e}")
        return False


def check_database_setup():
    """Check database configuration."""
    print("\n9. Database Setup (Optional):")
    
    try:
        import psycopg2
        print("   ✓ psycopg2 installed")
        
        try:
            from database import DatabaseConfig
            config = DatabaseConfig.from_env()
            print(f"   ℹ Database configured: {config.database} @ {config.host}")
        except Exception as e:
            print(f"   ℹ Database not configured: {e}")
            
    except ImportError:
        print("   ○ PostgreSQL support not installed (optional)")
    
    return True


def print_recommendations(results):
    """Print recommendations based on validation results."""
    print_header("RECOMMENDATIONS")
    
    if not results['required_packages']:
        print("\n⚠ Install required packages:")
        print("  pip install -r requirements.txt")
    
    if not results['configuration']:
        print("\n⚠ Set up configuration:")
        print("  1. Copy config.py.example to config.py")
        print("  2. Add your Alpaca API credentials")
        print("  3. Update other settings as needed")
    
    if not results['project_structure']:
        print("\n⚠ Some project files/directories are missing")
        print("  Ensure you have the complete repository")
    
    print("\n📚 Next Steps:")
    print("  1. Run a test backtest: python main.py")
    print("  2. Run tests: python tests/test_integration.py")
    print("  3. Check advanced features: python demo_advanced_metrics.py")
    
    print("\n💡 Optional Enhancements:")
    print("  - Install scipy for advanced risk metrics")
    print("  - Install matplotlib for charts and visualizations")
    print("  - Install psycopg2-binary for database support")
    print("  - Set up PostgreSQL for storing results")


def main():
    """Main validation function."""
    print_header("BACKTESTING SYSTEM VALIDATION")
    print("Checking your environment and setup...")
    
    results = {
        'python_version': check_python_version(),
        'required_packages': check_required_packages(),
        'optional_packages': check_optional_packages(),
        'project_structure': check_project_structure(),
        'configuration': check_configuration(),
        'data_directory': check_data_directory(),
        'imports': test_imports(),
        'functionality': run_simple_test(),
        'database': check_database_setup()
    }
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    critical_checks = ['python_version', 'required_packages', 'project_structure', 'imports']
    critical_passed = all(results[check] for check in critical_checks)
    
    if critical_passed:
        print("\n✅ SYSTEM READY!")
        print("All critical components are properly installed and configured.")
        
        if results['functionality']:
            print("\n✨ Basic functionality test passed!")
            print("You can now run backtests with: python main.py")
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("Some critical components need attention.")
    
    print("\nValidation Results:")
    for check, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check.replace('_', ' ').title()}")
    
    # Recommendations
    print_recommendations(results)
    
    return critical_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)