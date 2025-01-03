from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import requests

# Configuration constants
AILIVE_SECRET_APIKEY = "EXAMPLEKEY-zero-walking"
SAVE_INTERVAL = 500_000  # Save every 500,000 steps
TOTAL_TIMESTEPS = 10_000_000  # Total steps for training
VIEW_LOGS = False
RENDER_MODE = False  # Set to True to visualize the environment

# Extract agent and skill names from API key
try:
    agent_name, skill_name = AILIVE_SECRET_APIKEY.split("-")[1:]
except ValueError:
    raise ValueError("Invalid API key format. Expected format: <prefix>-<agent_name>-<skill_name>.")

# Paths for saving TensorBoard logs and models
TENSORBOARD_PATH = os.path.join(".", "sessions", agent_name, skill_name, "tensorboard")
MODELS_PATH = os.path.join(".", "sessions", agent_name, skill_name, "models")

# Ensure directories exist
os.makedirs(TENSORBOARD_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

def get_presigned_url():
    """
    Fetch a pre-signed URL from the API for uploading the model.
    """
    url = "https://api.ailive.co/v1/upload/sign_url"
    payload = {"api": AILIVE_SECRET_APIKEY}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()

        if not response_data.get("success", False):
            print(f"Error fetching pre-signed URL: {response_data.get('message', 'Unknown error')}")
            return None

        return response_data.get("url")
    except requests.RequestException as e:
        print(f"Request error while fetching pre-signed URL: {e}")
        return None

def upload_model(file_path):
    """
    Upload the model to the server using a pre-signed URL.
    """
    presigned_url = get_presigned_url()
    if not presigned_url:
        print("Failed to get a pre-signed URL. Skipping upload.")
        return False

    try:
        with open(file_path, "rb") as file:
            response = requests.put(presigned_url, data=file)
            response.raise_for_status()
            print(f"Model uploaded successfully to {presigned_url}.")
            return True
    except requests.RequestException as e:
        print(f"Error during model upload: {e}")
        return False

def save_model(model, step_count):
    """
    Save the model locally and upload it to the server.
    """
    file_name = f"{step_count}.zip"
    file_path = os.path.join(MODELS_PATH, file_name)

    model.save(file_path)
    print(f"Model saved locally: {file_path}")

    if not upload_model(file_path):
        print("Upload failed. Model saved locally for future attempts.")

def load_latest_model(model):
    """
    Load the latest saved model if available and return the starting step count.
    """
    saved_models = [f for f in os.listdir(MODELS_PATH) if f.endswith(".zip")]

    if not saved_models:
        print("No pre-trained model found. Starting training from scratch.")
        return 0

    latest_model = max(saved_models, key=lambda f: int(f.split(".")[0]))
    model_path = os.path.join(MODELS_PATH, latest_model)

    model.set_parameters(model_path)
    step_count = int(latest_model.split(".")[0])
    print(f"Resumed training from saved model: {model_path}. Starting at step {step_count}.")
    return step_count

def main():
    """
    Main function to set up the environment, train the model, and save progress.
    """
    # Create the environment with optional rendering
    if RENDER_MODE:
        print("Warning: Enabling visualization will slow down training.")
        env = make_vec_env("Humanoid-v5", env_kwargs={"render_mode": "human"})
    else:
        env = make_vec_env("Humanoid-v5")

    # Initialize the PPO model
    print(f"Logging TensorBoard data to: {TENSORBOARD_PATH}")
    model = PPO("MlpPolicy", env, verbose=VIEW_LOGS, tensorboard_log=TENSORBOARD_PATH)

    # Load the latest model if available
    steps_trained = load_latest_model(model)

    print(f"Starting training for a total of {TOTAL_TIMESTEPS} timesteps.")
    while steps_trained < TOTAL_TIMESTEPS:
        model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False, tb_log_name=skill_name)
        steps_trained += SAVE_INTERVAL
        save_model(model, steps_trained)

    env.close()
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
