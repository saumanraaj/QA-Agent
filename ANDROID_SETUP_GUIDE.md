# 🚀 Android Testing Setup Guide

This guide will help you set up the complete Android testing environment for the QA Agent with frame-by-frame UI rendering.

## 📋 Prerequisites

- ✅ Android Studio installed
- ✅ Android SDK installed
- ✅ Environment variables set up

## 🛠️ Step-by-Step Setup

### Step 1: Create Android Virtual Device (AVD)

1. **Open Android Studio**
   ```bash
   open -a "Android Studio"
   ```

2. **Create AVD via Android Studio GUI:**
   - Go to **Tools** > **AVD Manager**
   - Click **Create Virtual Device**
   - Select **Pixel 6** (or any Pixel device)
   - Select **API Level 33** (Tiramisu)
   - Name it **Pixel_6_API_33**
   - Click **Finish**

3. **Or create via command line:**
   ```bash
   # Install system image first
   sdkmanager "system-images;android-33;google_apis;x86_64"
   
   # Create AVD
   avdmanager create avd \
     --name Pixel_6_API_33 \
     --package "system-images;android-33;google_apis;x86_64" \
     --device "pixel_6"
   ```

### Step 2: Launch Android Emulator

```bash
# Launch the emulator
emulator -avd Pixel_6_API_33 -no-snapshot -grpc 8554
```

**Expected output:**
```
emulator: Android emulator version 36.0.0-13206524
emulator: Found AVD name 'Pixel_6_API_33'
emulator: Found AVD target architecture: x86_64
emulator: argv[0]: "/Users/saumanraaj/Library/Android/sdk/emulator/emulator"
emulator: argv[1]: "-avd"
emulator: argv[2]: "Pixel_6_API_33"
emulator: argv[3]: "-no-snapshot"
emulator: argv[4]: "-grpc"
emulator: argv[5]: "8554"
```

### Step 3: Verify Emulator Connection

```bash
# Check if emulator is running
adb devices
```

**Expected output:**
```
List of devices attached
emulator-5554	device
```

### Step 4: Install Android World Dependencies

```bash
# Navigate to android_world directory
cd android_world

# Install requirements
pip install -r requirements.txt

# Install Android World
python setup.py install
```

### Step 5: Test the QA Agent

```bash
# Return to QA testing directory
cd ..

# Test the QA Agent
python main.py
```

## 🧪 Testing Commands

### Quick Test
```bash
# Test if emulator is ready
python test_android_qa.py
```

### Full Test with Screenshots
```bash
# Run the main application
python main.py
```

### Check Generated Files
```bash
# View screenshots
ls -la traces/settings_wifi/

# View test logs
ls -la logs/
```

## 📁 Expected File Structure

After running the tests, you should see:

```
QA testing /
├── traces/
│   └── settings_wifi/
│       ├── step_0.png
│       ├── step_1.png
│       └── ...
├── logs/
│   ├── test_trace_*.json
│   └── test_summary_*.json
├── main.py
├── test_android_qa.py
└── setup_android_testing.py
```

## 🔧 Troubleshooting

### Issue: "adb not found"
```bash
# Add to your shell profile (~/.zshrc or ~/.bash_profile)
export ANDROID_HOME=$HOME/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator

# Reload shell
source ~/.zshrc
```

### Issue: "Emulator not starting"
```bash
# Check available AVDs
emulator -list-avds

# If no AVDs, create one via Android Studio GUI
open -a "Android Studio"
```

### Issue: "Android World installation failed"
```bash
# Check if android_world directory exists
ls -la android_world/

# If not, clone it
git clone https://github.com/google-research/android_world.git
```

### Issue: "Connection failed"
```bash
# Check emulator status
adb devices

# Restart emulator if needed
adb kill-server
adb start-server
emulator -avd Pixel_6_API_33 -no-snapshot -grpc 8554
```

## 🎯 Success Criteria

✅ **Android SDK installed and configured**  
✅ **ADB and emulator commands working**  
✅ **Pixel_6_API_33 AVD created**  
✅ **Emulator running and connected**  
✅ **Android World installed**  
✅ **QA Agent runs without errors**  
✅ **Screenshots captured in traces/**  
✅ **Test logs generated in logs/**  

## 🚀 Next Steps

1. **Run the setup script:**
   ```bash
   python setup_android_testing.py
   ```

2. **Test the complete system:**
   ```bash
   python main.py
   ```

3. **Check results:**
   ```bash
   ls -la traces/settings_wifi/
   ls -la logs/
   ```

4. **View test traces:**
   ```bash
   cat logs/test_trace_*.json | jq '.test_summary'
   ```

## 📊 Expected Output

When successful, you should see:

```
=== Agent-S Multi-Agent QA System (OTA AndroidEnv) ===
Task: Test turning Wi-Fi on and off
Task Name: settings_wifi

Initializing Android environment...
Environment initialized successfully

=== Planning Phase ===
Generated 3 subgoals:
  1. Open Settings app
  2. Navigate to Wi-Fi settings
  3. Toggle Wi-Fi on and off

=== Execution Phase ===
Step 1/3: Open Settings app
  PASS: Settings app opened successfully

Step 2/3: Navigate to Wi-Fi settings
  PASS: Wi-Fi settings found

Step 3/3: Toggle Wi-Fi on and off
  PASS: Wi-Fi toggled successfully

=== Test Results ===
Passed: 3/3 steps
Success Rate: 100.0%
Screenshots captured: 4
```

## 🎉 You're Ready!

Once you complete this setup, your QA Agent will be able to:

- ✅ Connect to Android emulator
- ✅ Execute real Android tasks
- ✅ Capture frame-by-frame screenshots
- ✅ Generate detailed test traces
- ✅ Support QualGent challenge requirements

Happy testing! 🚀 