# 🤖 AILIVE - Local Training with Stable-Baselines3

This project trains a reinforcement learning agent using Proximal Policy Optimization (PPO) from the Stable-Baselines3 library. The project includes automatic model saving, uploading models to a remote server, and resuming training from previously saved models.

---

## Features
- Uses PPO algorithm from Stable-Baselines3
- Supports TensorBoard logging
- Automatically saves models at regular intervals
- Uploads saved models to a remote server via pre-signed URLs
- Resumes training from the latest saved model

---

## Prerequisites
- Python 3.7 or later
- `pip` for package management

---

## Installation

1. Clone the repository:
   ```bash 
   git clone https://github.com/ailiveco/local-trainings.git
   cd local-trainings
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Update the following constants in `main.py` as needed:

- **API Key**: Replace `AILIVE_SECRET_APIKEY` with your valid API key.
  ```python
  AILIVE_SECRET_APIKEY = "EXAMPLEKEY-zero-walking"
  ```

- **Save Interval**: Adjust the frequency of model saving.
  ```python
  SAVE_INTERVAL = 500_000  # Save every 500,000 steps
  ```

- **Total Training Timesteps**: Set the total number of timesteps for training.
  ```python
  TOTAL_TIMESTEPS = 10_000_000  # Train for 10 million steps
  ```

---

## Usage

1. Run the main script to start training:
   ```bash
   python main.py
   ```

2. TensorBoard logs are saved in the specified directory. To visualize logs:
   ```bash
   tensorboard --logdir=./sessions/<agent_name>/<skill_name>/tensorboard
   ```

3. Models are saved in the `models/` directory inside the session folder.

---

## Project Structure
```
.
├── main.py                  # Main script for training the agent
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation
└── sessions/                # Directory for logs and saved models
    └── <agent_name>/        # Agent-specific folder
        └── <skill_name>/    # Skill-specific folder
            ├── tensorboard/ # TensorBoard logs
            └── models/      # Saved models
```

---

## Key Functions

### `get_presigned_url()`
Fetches a pre-signed URL for uploading models to a remote server.

### `upload_model(file_path)`
Uploads a model file to the server using the pre-signed URL.

### `save_model(model, step_count)`
Saves the model locally and uploads it to the server.

### `load_latest_model(model)`
Loads the latest saved model for resuming training.

### `main()`
Sets up the environment, initializes the PPO model, and manages the training loop.

---

## Notes
- Ensure your API key is valid and matches the expected format.
- TensorBoard must be installed to use logging features.
- The `Humanoid-v5` environment is used as an example; you can replace it with any supported environment.

---

## License
This project is licensed under the MIT License.
