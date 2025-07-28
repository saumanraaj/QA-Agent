# Android in the Wild Integration with Multi-Agent QA System

This project integrates the [Android in the Wild (AitW)](https://github.com/google-research/google-research/tree/master/android_in_the_wild) dataset with the multi-agent QA architecture to enhance training, evaluation, and robustness of the system.

## Overview

The Android in the Wild dataset contains:
- **Screen recordings and UI traces** from real user sessions across thousands of Android apps
- **Semantic and visual diversity** covering notifications, modals, errors, dialogs, dark mode, and inconsistent layouts
- **Real-world complexity** useful for training and evaluating agents on UI distribution shifts

## Features

### 1. Real Data Processing
- Downloads actual android_in_the_wild data from Google Cloud Storage
- Processes TFRecord files into structured JSON format
- Extracts episodes, actions, UI annotations, and metadata

### 2. Task Prompt Generation
- Analyzes episode goals and context to generate clear task prompts
- Uses LLM to infer user intent from natural language instructions
- Calculates confidence and complexity scores for generated prompts

### 3. Multi-Agent Evaluation
- Simulates agent execution of tasks using the multi-agent QA system
- Compares agent performance with ground truth traces
- Evaluates accuracy, robustness, and generalization

### 4. Comprehensive Metrics
- **Accuracy**: Action type matching and coordinate precision
- **Robustness**: Ability to handle UI variations and complexity
- **Generalization**: Performance across different task types
- **Completion Rate**: Percentage of steps successfully completed

## Installation

### Prerequisites
```bash
# Install required dependencies
pip install -r android_in_the_wild_requirements.txt

# Clone the google-research repository (already done)
git clone https://github.com/google-research/google-research.git
```

### Setup
```bash
# Make scripts executable
chmod +x download_android_in_the_wild_data.py
chmod +x android_in_the_wild_qa_integration.py
```

## Usage

### Step 1: Download and Process Data

```bash
python download_android_in_the_wild_data.py
```

This script will:
- Download real android_in_the_wild data from Google Cloud Storage
- Process TFRecord files into structured JSON format
- Generate dataset summaries and statistics
- Save processed data to `./android_in_the_wild_data/`

### Step 2: Run QA Evaluation

```bash
python android_in_the_wild_qa_integration.py
```

This script will:
- Load processed episodes from different datasets
- Generate task prompts using LLM analysis
- Simulate agent execution of tasks
- Compare with ground truth traces
- Generate comprehensive evaluation reports

### Step 3: Analyze Results

The evaluation generates:
- **JSON Reports**: Detailed evaluation metrics and results
- **Visualizations**: Histograms and correlation heatmaps
- **Summary Statistics**: Aggregate performance metrics

## Dataset Information

### Available Datasets

1. **GoogleApps** (625,542 episodes)
   - High-level tasks involving Google applications
   - Examples: "turn off javascript in the chrome app"

2. **Install** (25,760 episodes)
   - App installation and login tasks
   - Examples: "open app 'DoorDash - Food Delivery' (install if not already installed)"

3. **WebShopping** (28,061 episodes)
   - E-commerce shopping tasks
   - Examples: "Look up the best rated coffee maker on Lowe's"

4. **General** (9,476 episodes)
   - Miscellaneous tasks and 3rd party apps
   - Examples: "Open a new Chrome private window"

5. **Single** (26,303 episodes)
   - Single-step tasks manually annotated
   - Examples: "Add to cart", "Go to ebay search bar and search lg ultragear"

### Action Types

- **DUAL_POINT** (4): Clicks and scrolls with touch/lift coordinates
- **TYPE** (3): Text input actions
- **PRESS_BACK** (5): Back button press
- **PRESS_HOME** (6): Home button press
- **PRESS_ENTER** (7): Enter key press
- **STATUS_TASK_COMPLETE** (10): Task completed successfully
- **STATUS_TASK_IMPOSSIBLE** (11): Task impossible to complete

## Evaluation Metrics

### Accuracy Metrics
- **Action Matching**: Percentage of correct action types
- **Coordinate Accuracy**: Precision of touch coordinates for dual-point actions
- **Overall Accuracy**: Combined score of action and coordinate accuracy

### Robustness Metrics
- **UI Variation Handling**: Performance across different screen layouts
- **Error Recovery**: Ability to handle unexpected UI states
- **Complexity Adaptation**: Performance on tasks with varying complexity

### Generalization Metrics
- **Cross-Dataset Performance**: Performance across different datasets
- **Task Type Diversity**: Performance on different types of tasks
- **Device Adaptation**: Performance across different device types

## Example Output

### Dataset Summary
```
=== GOOGLE_APPS DATASET SUMMARY ===
Total Episodes: 5
Total Steps: 47
Average Steps per Episode: 9.40
Unique Activities: 8
Device Types: ['pixel_4']
Android Versions: [30]
Action Type Distribution: {'TYPE': 5, 'DUAL_POINT': 42}
```

### Evaluation Report
```
=== GOOGLE_APPS QA EVALUATION REPORT ===
Total Episodes: 3
Average Accuracy: 0.823 ± 0.156
Average Action Matching: 0.867 ± 0.133
Average Completion Rate: 0.733 ± 0.231
Average Robustness: 0.789 ± 0.145
Average Generalization: 0.756 ± 0.178
Task Prompt Confidence: 0.650
Task Complexity: 0.567
```

## Integration with Android World

The system is designed to integrate with the existing android_world environment:

```python
# Example integration with android_world
from android_world import AndroidWorld
from android_in_the_wild_qa_integration import AndroidInTheWildQAEvaluator

# Initialize android_world environment
android_env = AndroidWorld()

# Run evaluation with real environment
evaluator = AndroidInTheWildQAEvaluator()
evaluator.run_evaluation_with_android_world(android_env, dataset_name='google_apps')
```

## File Structure

```
├── android_in_the_wild_integration.py      # Main integration script
├── download_android_in_the_wild_data.py    # Data download and processing
├── android_in_the_wild_qa_integration.py   # QA evaluation system
├── android_in_the_wild_requirements.txt    # Dependencies
├── README_ANDROID_IN_THE_WILD.md          # This file
├── android_in_the_wild_data/              # Processed data directory
│   ├── google_apps_processed_episodes.json
│   ├── web_shopping_processed_episodes.json
│   ├── general_processed_episodes.json
│   └── summaries/
└── google-research/                       # Cloned repository
    └── android_in_the_wild/
        ├── README.md
        ├── demo.ipynb
        ├── action_matching.py
        ├── action_type.py
        └── visualization_utils.py
```

## Advanced Usage

### Custom Evaluation
```python
from android_in_the_wild_qa_integration import AndroidInTheWildQAEvaluator

# Initialize evaluator
evaluator = AndroidInTheWildQAEvaluator()

# Run custom evaluation
evaluator.run_evaluation(
    dataset_name='web_shopping',
    max_episodes=10,
    custom_metrics=['custom_metric_1', 'custom_metric_2']
)
```

### Batch Processing
```python
# Process multiple datasets
datasets = ['google_apps', 'web_shopping', 'general', 'install', 'single']

for dataset in datasets:
    evaluator.run_evaluation(dataset, max_episodes=5)
```

### Custom Task Prompt Generation
```python
# Generate custom task prompts
episode = evaluator.load_processed_episodes('google_apps')[0]
task_prompt = evaluator.generate_task_prompt(episode)
print(f"Generated task: {task_prompt['generated_task']}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure google-research is cloned
   git clone https://github.com/google-research/google-research.git
   
   # Add to Python path
   export PYTHONPATH="${PYTHONPATH}:./google-research"
   ```

2. **TensorFlow Issues**
   ```bash
   # Install compatible TensorFlow version
   pip install tensorflow==2.8.0
   ```

3. **Google Cloud Storage Access**
   ```bash
   # Install Google Cloud SDK
   pip install google-cloud-storage
   
   # Authenticate (if needed)
   gcloud auth application-default login
   ```

### Performance Optimization

1. **Memory Usage**
   - Limit `max_episodes` for large datasets
   - Use `max_files` parameter in data downloader

2. **Processing Speed**
   - Use multiprocessing for batch processing
   - Cache processed episodes to avoid reprocessing

## Contributing

To contribute to this integration:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the Apache License 2.0, same as the original android_in_the_wild dataset.

## References

- [Android in the Wild Paper](https://arxiv.org/abs/2307.10088)
- [Google Research Repository](https://github.com/google-research/google-research/tree/master/android_in_the_wild)
- [Android World Environment](https://github.com/google-research/android_world)

## Citation

If you use this integration in your research, please cite:

```bibtex
@article{android_in_the_wild_2023,
  title={Android in the Wild: A Large-Scale Dataset for Android Device Control},
  author={...},
  journal={arXiv preprint arXiv:2307.10088},
  year={2023}
}
``` 