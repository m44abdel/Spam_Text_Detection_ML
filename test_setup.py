#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly
Run this before starting the notebook
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn'),
        ('nltk', 'nltk'),
        ('sklearn', 'sklearn'),
        ('tensorflow', 'tensorflow'),
    ]
    
    print("Testing package imports...")
    print("=" * 60)
    
    failed = []
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✓ {package_name:20s} - OK")
        except ImportError as e:
            print(f"✗ {package_name:20s} - FAILED")
            failed.append(package_name)
    
    print("=" * 60)
    
    if failed:
        print(f"\nFailed to import: {', '.join(failed)}")
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(failed)}")
        return False
    else:
        print("\nAll packages imported successfully!")
        
        # Test NLTK data
        print("\nDownloading required NLTK data...")
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            print("✓ NLTK data downloaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: NLTK download failed: {e}")
        
        return True

def check_dataset():
    """Check if the dataset file exists"""
    import os
    print("\nChecking for dataset file...")
    print("=" * 60)
    
    if os.path.exists('SMSSpamCollection'):
        # Check file size
        size = os.path.getsize('SMSSpamCollection')
        print(f"✓ Dataset file found (Size: {size:,} bytes)")
        
        # Count lines
        with open('SMSSpamCollection', 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"✓ Dataset contains {line_count:,} messages")
        return True
    else:
        print("✗ Dataset file 'SMSSpamCollection' not found!")
        print("Please download from: https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
        return False

if __name__ == "__main__":
    print("SMS Spam Detection - Setup Verification")
    print("=" * 60)
    print()
    
    imports_ok = test_imports()
    dataset_ok = check_dataset()
    
    print("\n" + "=" * 60)
    if imports_ok and dataset_ok:
        print("🎉 Setup complete! You're ready to run the notebook.")
        print("\nStart Jupyter with: jupyter notebook spam_detection.ipynb")
        sys.exit(0)
    else:
        print("⚠️  Setup incomplete. Please resolve the issues above.")
        sys.exit(1)

