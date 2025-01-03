import json
import logging
from openai import OpenAI
from src.config import CFG


def load_api_key(api_key_path):
    """
    Load the OpenAI API key from a JSON file specified in the configuration.

    Parameters:
        api_key_path : path to api key.

    Returns:
        OpenAI: OpenAI client initialized with the API key.
    """
    with open(api_key_path, "r") as f:
        api_key = json.load(f)["key"]
    logging.info("API key loaded successfully.")
    return OpenAI(api_key=api_key)


def upload_file(client, file_path, purpose="fine-tune"):
    """
    Upload a file to OpenAI for a specific purpose.

    Parameters:
        client (OpenAI): OpenAI client.
        file_path (Path): Path to the file to upload.
        purpose (str): Purpose of the file upload (default: "fine-tune").

    Returns:
        Response: File upload response from OpenAI.
    """
    logging.info(f"Uploading file: {file_path} with purpose: {purpose}")
    with open(file_path, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    logging.info(f"File uploaded successfully: {response}")
    return response


def create_fine_tuning_job(client, training_file_id, validation_file_id, model, epochs):
    """
    Create a fine-tuning job using the provided training and validation files.

    Parameters:
        client (OpenAI): OpenAI client.
        training_file_id (str): ID of the training file.
        validation_file_id (str): ID of the validation file.
        model (str): Model to fine-tune.
        epochs (int): Number of epochs for fine-tuning.

    Returns:
        Response: Fine-tuning job creation response.
    """
    logging.info("Creating fine-tuning job.")
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        hyperparameters={"n_epochs": epochs},
        validation_file=validation_file_id
    )
    logging.info(f"Fine-tuning job created: {response}")
    return response


def list_fine_tuning_jobs(client):
    """
    List all fine-tuning jobs.

    Parameters:
        client (OpenAI): OpenAI client.

    Returns:
        Response: List of fine-tuning jobs.
    """
    logging.info("Fetching list of fine-tuning jobs.")
    response = client.fine_tuning.jobs.list()
    logging.info(f"Fine-tuning jobs: {response}")
    return response


def main():
    logging.basicConfig(level=logging.INFO)
    client = load_api_key(CFG.root.joinpath("openai_key.json"))

    train_file_response = upload_file(client, CFG.output_path.joinpath("train.json"))
    val_file_response = upload_file(client, CFG.output_path.joinpath("val.json"))

    fine_tuning_response = create_fine_tuning_job(
        client,
        training_file_id=train_file_response.id,
        validation_file_id=val_file_response.id,
        model=CFG.model,
        epochs=CFG.epoch_num
    )

    logging.info(f"Fine-tuning job response: {fine_tuning_response}")
    jobs_list = list_fine_tuning_jobs(client)
    logging.info(f"Fine-tuning jobs list: {jobs_list}")


if __name__ == "__main__":
    main()
