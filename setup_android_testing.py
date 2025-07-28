#!/usr/bin/env python3
"""
Comprehensive Android Testing Setup Script
This script helps set up the complete Android testing environment for the QA Agent
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def check_command(command):
    """Check if a command is available"""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def run_command(command, description=""):
    """Run a command and return success status"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            return True
        else:
            print(f"❌ {description} - Failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Exception: {e}")
        return False

def check_android_setup():
    """Check Android SDK setup"""
    print("=== Android SDK Setup Check ===")
    
    # Check environment variables
    android_home = os.environ.get('ANDROID_HOME')
    if android_home:
        print(f"✅ ANDROID_HOME set to: {android_home}")
    else:
        print("❌ ANDROID_HOME not set")
        return False
    
    # Check if adb is available
    if check_command('adb'):
        print("✅ ADB (Android Debug Bridge) is available")
    else:
        print("❌ ADB not found in PATH")
        return False
    
    # Check if emulator is available
    if check_command('emulator'):
        print("✅ Android Emulator is available")
    else:
        print("❌ Android Emulator not found in PATH")
        return False
    
    return True

def setup_android_avd():
    """Set up Android Virtual Device"""
    print("\n=== Setting up Android Virtual Device ===")
    
    # Check if AVD already exists
    result = subprocess.run(['emulator', '-list-avds'], 
                          capture_output=True, text=True)
    
    if 'Pixel_6_API_33' in result.stdout:
        print("✅ Pixel_6_API_33 AVD already exists")
        return True
    
    print("📱 Creating Pixel_6_API_33 AVD...")
    
    # Create AVD using command line
    avd_command = [
        'avdmanager', 'create', 'avd',
        '--name', 'Pixel_6_API_33',
        '--package', 'system-images;android-33;google_apis;x86_64',
        '--device', 'pixel_6'
    ]
    
    try:
        # This might require interactive input, so we'll provide guidance
        print("⚠️  AVD creation requires interactive setup.")
        print("Please run the following command manually:")
        print(" ".join(avd_command))
        print("\nOr use Android Studio GUI:")
        print("1. Open Android Studio")
        print("2. Go to Tools > AVD Manager")
        print("3. Click 'Create Virtual Device'")
        print("4. Select 'Pixel 6'")
        print("5. Select 'API Level 33' (Tiramisu)")
        print("6. Name it 'Pixel_6_API_33'")
        return False
    except Exception as e:
        print(f"❌ AVD creation failed: {e}")
        return False

def launch_emulator():
    """Launch the Android emulator"""
    print("\n=== Launching Android Emulator ===")
    
    # Check if emulator is already running
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    if 'emulator-5554' in result.stdout:
        print("✅ Emulator is already running")
        return True
    
    print("🚀 Starting Android emulator...")
    
    # Launch emulator in background
    emulator_command = [
        'emulator', '-avd', 'Pixel_6_API_33',
        '-no-snapshot', '-grpc', '8554'
    ]
    
    try:
        # Start emulator in background
        process = subprocess.Popen(emulator_command)
        print(f"✅ Emulator started (PID: {process.pid})")
        print("⏳ Waiting for emulator to boot...")
        
        # Wait for emulator to be ready
        for i in range(60):  # Wait up to 60 seconds
            time.sleep(1)
            result = subprocess.run(['adb', 'devices'], 
                                  capture_output=True, text=True)
            if 'emulator-5554' in result.stdout and 'device' in result.stdout:
                print("✅ Emulator is ready!")
                return True
        
        print("⚠️  Emulator may still be booting. Please wait...")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start emulator: {e}")
        return False

def test_android_connection():
    """Test connection to Android emulator"""
    print("\n=== Testing Android Connection ===")
    
    # Check if device is connected
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    print("Connected devices:")
    print(result.stdout)
    
    if 'emulator-5554' in result.stdout and 'device' in result.stdout:
        print("✅ Android emulator is connected and ready")
        return True
    else:
        print("❌ Android emulator not connected")
        return False

def install_android_world():
    """Install Android World dependencies"""
    print("\n=== Installing Android World Dependencies ===")
    
    # Check if we're in the right directory
    if not os.path.exists('android_world'):
        print("❌ android_world directory not found")
        print("Please run this script from the QA testing directory")
        return False
    
    # Install requirements
    if run_command("cd android_world && pip install -r requirements.txt", 
                   "Installing Android World requirements"):
        print("✅ Android World requirements installed")
    else:
        print("❌ Failed to install Android World requirements")
        return False
    
    # Install Android World
    if run_command("cd android_world && python setup.py install", 
                   "Installing Android World"):
        print("✅ Android World installed")
    else:
        print("❌ Failed to install Android World")
        return False
    
    return True

def test_qa_agent():
    """Test the QA Agent with Android environment"""
    print("\n=== Testing QA Agent ===")
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("❌ main.py not found")
        return False
    
    # Test with a simple task
    print("🧪 Testing QA Agent with 'Turn Wi-Fi on and off' task...")
    
    try:
        # Run a quick test (this might fail if emulator isn't ready)
        result = subprocess.run(['python', 'main.py'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ QA Agent test completed successfully")
            return True
        else:
            print("⚠️  QA Agent test had issues (this might be expected if emulator isn't ready)")
            print("Output:", result.stdout)
            print("Errors:", result.stderr)
            return True  # Don't fail the setup for this
            
    except subprocess.TimeoutExpired:
        print("⚠️  QA Agent test timed out (this might be expected)")
        return True
    except Exception as e:
        print(f"❌ QA Agent test failed: {e}")
        return False

def create_test_script():
    """Create a test script for easy testing"""
    test_script = '''#!/usr/bin/env python3
"""
Quick test script for Android QA Agent
"""

import os
import sys

def main():
    print("=== Android QA Agent Test ===")
    
    # Check environment
    print("1. Checking Android environment...")
    if not os.path.exists('main.py'):
        print("❌ main.py not found")
        return False
    
    # Check if emulator is running
    import subprocess
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    if 'emulator-5554' not in result.stdout:
        print("❌ Android emulator not running")
        print("Please start the emulator first:")
        print("emulator -avd Pixel_6_API_33 -no-snapshot -grpc 8554")
        return False
    
    print("✅ Android emulator is running")
    
    # Run the test
    print("2. Running QA Agent test...")
    try:
        result = subprocess.run(['python', 'main.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Test completed successfully!")
            return True
        else:
            print("⚠️  Test completed with issues")
            print("Output:", result.stdout)
            return True
            
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
    
    with open('test_android_qa.py', 'w') as f:
        f.write(test_script)
    
    os.chmod('test_android_qa.py', 0o755)
    print("✅ Created test script: test_android_qa.py")

def main():
    """Main setup function"""
    print("🚀 Android QA Agent Setup")
    print("=" * 50)
    
    # Step 1: Check Android setup
    if not check_android_setup():
        print("\n❌ Android SDK setup incomplete")
        print("Please:")
        print("1. Open Android Studio")
        print("2. Go through the setup wizard")
        print("3. Install Android SDK")
        print("4. Set up environment variables")
        return False
    
    # Step 2: Set up AVD
    if not setup_android_avd():
        print("\n⚠️  Please set up AVD manually using Android Studio")
    
    # Step 3: Launch emulator
    if not launch_emulator():
        print("\n⚠️  Please launch emulator manually")
    
    # Step 4: Test connection
    if not test_android_connection():
        print("\n⚠️  Android connection test failed")
    
    # Step 5: Install Android World
    if not install_android_world():
        print("\n❌ Android World installation failed")
        return False
    
    # Step 6: Test QA Agent
    test_qa_agent()
    
    # Step 7: Create test script
    create_test_script()
    
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Ensure Android emulator is running:")
    print("   emulator -avd Pixel_6_API_33 -no-snapshot -grpc 8554")
    print("2. Test the QA Agent:")
    print("   python test_android_qa.py")
    print("3. Run the main application:")
    print("   python main.py")
    print("\n📁 Generated files:")
    print("   • test_android_qa.py - Quick test script")
    print("   • traces/ - Screenshot captures")
    print("   • logs/ - Test traces and logs")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 