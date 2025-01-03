import json
from loguru import logger
import pandas as pd
import tiktoken
from pathlib import Path

from src.config import CFG


def load_and_clean_data(csv_path):
    """
    Load the medical reports from a CSV file and drop rows with empty reports.

    Parameters:
        csv_path (str or Path): Path to the input CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info("Loading medical reports from: %s", csv_path)
    medical_reports = pd.read_csv(csv_path)

    logger.info("Dropping rows with empty reports.")
    initial_count = len(medical_reports)
    medical_reports.dropna(subset=["report"], inplace=True)
    logger.info("Dropped %d rows with empty reports.", initial_count - len(medical_reports))

    return medical_reports


def balance_data(data, sample_size, random_state):
    """
    Balance the dataset by sampling an equal number of samples per specialty.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        sample_size (int): Number of samples per specialty.
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: Balanced DataFrame.
    """
    logger.info(f"Balancing dataset: Sampling {sample_size} samples per specialty.")
    return data.groupby("medical_specialty").sample(sample_size, random_state=random_state)


def split_data(grouped_data, val_test_sample, random_state):
    """
    Split the data into training, validation, and testing sets.

    Parameters:
        grouped_data (pd.DataFrame): Balanced DataFrame.
        val_test_sample (int): Number of samples per specialty for validation and testing.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: DataFrames for training, validation, and testing.
    """
    logger.info("Creating validation and test datasets.")
    val_test_data = grouped_data.groupby("medical_specialty").sample(val_test_sample, random_state=random_state)

    val = val_test_data.groupby("medical_specialty").head(val_test_sample // 2)
    test = val_test_data.groupby("medical_specialty").tail(val_test_sample // 2)
    train = grouped_data[~grouped_data.index.isin(val_test_data.index)]

    return train, val, test


def calculate_token_stats(train_data):
    """
    Calculate token statistics for the training data.

    Parameters:
        train_data (pd.DataFrame): DataFrame containing the training data.

    Returns:
        dict: Token statistics including average, minimum, maximum, and total token counts.
    """
    logger.info("Counting tokens in training data.")
    encoding = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_string(string):
        return len(encoding.encode(string))

    report_lengths = train_data["report"].apply(num_tokens_from_string)
    stats = {
        "average": report_lengths.mean(),
        "minimum": report_lengths.min(),
        "maximum": report_lengths.max(),
        "total": report_lengths.sum()
    }
    logger.info(f"Token length statistics: {stats}")
    return stats


def estimate_fine_tuning_cost(token_stats, price_model=0.003):
    """
    Estimate the cost of fine-tuning.

    Parameters:
        token_stats (dict): Token statistics.
        price_model (float): Price per 1k tokens for fine-tuning.

    Returns:
        float: Estimated cost per epoch.
    """
    cost = token_stats["total"] * price_model / 1000
    return cost


def format_data(df, system_prompt):
    """
    Format the data into the JSONL format required for fine-tuning.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        system_prompt (str): System prompt for the fine-tuning task.

    Returns:
        list: List of formatted entries.
    """
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row["report"]},
                {"role": "assistant", "content": row["medical_specialty"]}
            ]
        })
    return formatted_data


def save_to_jsonl(data, output_path):
    """
    Save formatted data to a JSONL file.

    Parameters:
        data (list): Formatted data entries.
        output_path (str or Path): Path to the output JSONL file.
    """
    logger.info(f"Saving data to {output_path}")
    with open(output_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry))
            f.write("\n")


def preprocess_medical_reports(csv_path, random_state=42, sample_size=110, val_test_sample=10, output_dir="."):
    """
    Main function to preprocess medical reports.

    Parameters:
        csv_path (str or Path): Path to the input CSV file.
        random_state (int): Random state for reproducibility.
        sample_size (int): Number of samples per specialty for training.
        val_test_sample (int): Number of samples per specialty for validation and testing.
        output_dir (str or Path): Directory to save the output JSONL files.

    Returns:
        None
    """
    medical_reports = load_and_clean_data(csv_path)
    grouped_data = balance_data(medical_reports, sample_size, random_state)
    train, val, test = split_data(grouped_data, val_test_sample, random_state)

    train_data = format_data(train, CFG.system_prompt)
    val_data = format_data(val, CFG.system_prompt)
    test_data = format_data(test, CFG.system_prompt)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_to_jsonl(train_data, output_dir / "train.json")
    save_to_jsonl(val_data, output_dir / "val.json")
    save_to_jsonl(test_data, output_dir / "test.json")

    token_stats = calculate_token_stats(train)
    cost = estimate_fine_tuning_cost(token_stats)

    logger.info("Training token statistics: Average: %.2f, Min: %d, Max: %d, Total: %d",
                 token_stats["average"], token_stats["minimum"], token_stats["maximum"], token_stats["total"])
    logger.info(f"Estimated fine-tuning cost per epoch: {float(cost)} $")


if __name__ == '__main__':
    preprocess_medical_reports(CFG.data_path.joinpath("reports.csv"), random_state=42, output_dir="output")
