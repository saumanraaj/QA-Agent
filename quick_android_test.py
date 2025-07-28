#!/usr/bin/env python3
"""
Quick Android Setup Test
This script checks the current status of Android setup
"""

import os
import sys
import subprocess

def check_command(command, description):
    """Check if a command is available"""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ {description}: Available")
            return True
        else:
            print(f"❌ {description}: Not available")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"❌ {description}: Not found")
        return False

def check_environment():
    """Check environment variables"""
    print("=== Environment Check ===")
    
    android_home = os.environ.get('ANDROID_HOME')
    if android_home:
        print(f"✅ ANDROID_HOME: {android_home}")
    else:
        print("❌ ANDROID_HOME: Not set")
        return False
    
    return True

def check_tools():
    """Check Android tools"""
    print("\n=== Tools Check ===")
    
    adb_ok = check_command('adb', 'ADB (Android Debug Bridge)')
    emulator_ok = check_command('emulator', 'Android Emulator')
    
    return adb_ok and emulator_ok

def check_avds():
    """Check available AVDs"""
    print("\n=== AVD Check ===")
    
    try:
        result = subprocess.run(['emulator', '-list-avds'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            avds = result.stdout.strip().split('\n')
            if avds and avds[0]:
                print("✅ Available AVDs:")
                for avd in avds:
                    if avd:
                        print(f"   • {avd}")
                
                # Check if we have the recommended AVD
                if 'Pixel_6_API_33' in avds:
                    print("✅ Recommended AVD (Pixel_6_API_33) found!")
                    return True
                else:
                    print("⚠️  Recommended AVD (Pixel_6_API_33) not found")
                    print("   Please create it using Android Studio")
                    return False
            else:
                print("❌ No AVDs found")
                return False
        else:
            print("❌ Failed to list AVDs")
            return False
    except Exception as e:
        print(f"❌ Error checking AVDs: {e}")
        return False

def check_emulator_status():
    """Check if emulator is running"""
    print("\n=== Emulator Status ===")
    
    try:
        result = subprocess.run(['adb', 'devices'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Connected devices:")
            print(result.stdout)
            
            if 'emulator-5554' in result.stdout and 'device' in result.stdout:
                print("✅ Android emulator is running and connected!")
                return True
            else:
                print("❌ Android emulator not running")
                print("   To start: emulator -avd Pixel_6_API_33 -no-snapshot -grpc 8554")
                return False
        else:
            print("❌ Failed to check devices")
            return False
    except Exception as e:
        print(f"❌ Error checking emulator: {e}")
        return False

def check_android_world():
    """Check Android World installation"""
    print("\n=== Android World Check ===")
    
    if os.path.exists('android_world'):
        print("✅ android_world directory found")
        
        # Check if it's properly set up
        if os.path.exists('android_world/setup.py'):
            print("✅ Android World setup.py found")
            return True
        else:
            print("❌ Android World not properly installed")
            return False
    else:
        print("❌ android_world directory not found")
        print("   Please clone: git clone https://github.com/google-research/android_world.git")
        return False

def main():
    """Main test function"""
    print("🔍 Android Setup Status Check")
    print("=" * 40)
    
    # Check environment
    env_ok = check_environment()
    
    # Check tools
    tools_ok = check_tools()
    
    # Check AVDs
    avds_ok = check_avds()
    
    # Check emulator
    emulator_ok = check_emulator_status()
    
    # Check Android World
    android_world_ok = check_android_world()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Setup Status Summary:")
    print(f"   Environment: {'✅' if env_ok else '❌'}")
    print(f"   Tools: {'✅' if tools_ok else '❌'}")
    print(f"   AVDs: {'✅' if avds_ok else '❌'}")
    print(f"   Emulator: {'✅' if emulator_ok else '❌'}")
    print(f"   Android World: {'✅' if android_world_ok else '❌'}")
    
    all_ok = env_ok and tools_ok and avds_ok and emulator_ok and android_world_ok
    
    if all_ok:
        print("\n🎉 All checks passed! You're ready to test the QA Agent.")
        print("\nNext steps:")
        print("1. Run: python main.py")
        print("2. Check results in traces/ and logs/ directories")
    else:
        print("\n⚠️  Some checks failed. Please follow the setup guide:")
        print("   See: ANDROID_SETUP_GUIDE.md")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 