import json
import os
import random
import time
import psutil
import threading
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch as T


def read_json_as_dict(input_path: str) -> Dict:
    """
    Reads a JSON file and returns its content as a dictionary.
    If input_path is a directory, the first JSON file in the directory is read.
    If input_path is a file, the file is read.

    Args:
        input_path (str): The path to the JSON file or directory containing a JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.

    Raises:
        ValueError: If the input_path is neither a file nor a directory,
                    or if input_path is a directory without any JSON files.
    """
    if os.path.isdir(input_path):
        # Get all the JSON files in the directory
        json_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".json")
        ]

        # If there are no JSON files, raise a ValueError
        if not json_files:
            raise ValueError("No JSON files found in the directory")

        # Else, get the path of the first JSON file
        json_file_path = json_files[0]

    elif os.path.isfile(input_path):
        json_file_path = input_path
    else:
        raise ValueError("Input path is neither a file nor a directory")

    # Read the JSON file and return it as a dictionary
    with open(json_file_path, "r", encoding="utf-8") as file:
        json_data_as_dict = json.load(file)

    return json_data_as_dict


def read_csv_in_directory(file_dir_path: str) -> pd.DataFrame:
    """
    Reads a CSV file in the given directory path as a pandas dataframe and returns
    the dataframe.

    Args:
    - file_dir_path (str): The path to the directory containing the CSV file.

    Returns:
    - pd.DataFrame: The pandas dataframe containing the data from the CSV file.

    Raises:
    - FileNotFoundError: If the directory does not exist.
    - ValueError: If no CSV file is found in the directory or if multiple CSV files are
        found in the directory.
    """
    if not os.path.exists(file_dir_path):
        raise FileNotFoundError(f"Directory does not exist: {file_dir_path}")

    csv_files = [file for file in os.listdir(file_dir_path) if file.endswith(".csv")]

    if not csv_files:
        raise ValueError(f"No CSV file found in directory {file_dir_path}")

    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in directory {file_dir_path}.")

    csv_file_path = os.path.join(file_dir_path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    return df


def cast_time_col(data: pd.DataFrame, time_col: str, dtype: str) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): dataframe
        time_col (str): name of the time field
        dtype (str): type of time field ('INT', 'DATE', or 'DATETIME')

    Returns:
        pd.DataFrame: updated dataframe with time col appropriately cast
    """
    data = data.copy()
    if dtype == "INT":
        data[time_col] = data[time_col].astype(int)
    elif dtype in ["DATETIME", "DATE"]:
        data[time_col] = pd.to_datetime(data[time_col])
    else:
        raise ValueError(f"Invalid data type for time column: {dtype}")
    return data


def set_seeds(seed_value: int) -> None:
    """
    Set the random seeds for Python, NumPy, etc. to ensure
    reproducibility of results.

    Args:
        seed_value (int): The seed value to use for random
            number generation. Must be an integer.

    Returns:
        None
    """
    if isinstance(seed_value, int):
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        T.manual_seed(seed_value)
        T.cuda.manual_seed(seed_value)
        T.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
    else:
        raise ValueError(f"Invalid seed value: {seed_value}. Cannot set seeds.")


def split_train_val_by_series(
    data: pd.DataFrame, val_pct: float, id_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input time-series data into training and validation sets based on the
    percentage of series. All data points from the same series are kept together.

    Args:
        data (pd.DataFrame): The input data as a DataFrame.
        val_pct (float): The percentage of series to be used for the validation set.
        id_col (str): The name of the column containing series ids.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and
            validation sets as DataFrames.
    """
    # Ensure the percentage is between 0 and 1
    if not 0 <= val_pct <= 1:
        raise ValueError("val_pct must be between 0 and 1")

    # Get unique series IDs
    series_ids = data[id_col].unique()

    # Shuffle series IDs
    shuffled_series_ids = np.random.permutation(series_ids)

    # Calculate the split index
    split_idx = int(len(shuffled_series_ids) * (1 - val_pct))

    # Split series IDs into train and validation sets
    train_series_ids = shuffled_series_ids[:split_idx]
    val_series_ids = shuffled_series_ids[split_idx:]

    # Split the data based on series IDs
    train_data = data[data[id_col].isin(train_series_ids)]
    val_data = data[data[id_col].isin(val_series_ids)]

    return train_data, val_data


def save_dataframe_as_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Saves a pandas dataframe to a CSV file in the given directory path.
    Float values are saved with 4 decimal places.

    Args:
    - df (pd.DataFrame): The pandas dataframe to be saved.
    - file_path (str): File path and name to save the CSV file.

    Returns:
    - None

    Raises:
    - IOError: If an error occurs while saving the CSV file.
    """
    try:
        dataframe.to_csv(file_path, index=False, float_format="%.8f")
    except IOError as exc:
        raise IOError(f"Error saving CSV file: {exc}") from exc


def clear_files_in_directory(directory_path: str) -> None:
    """
    Clears all files in the given directory path.

    Args:
    - directory_path (str): The path to the directory containing the files
        to be cleared.

    Returns:
    - None
    """
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        os.remove(file_path)


def save_json(file_path_and_name: str, data: Any) -> None:
    """Save json to a path (directory + filename)"""
    with open(file_path_and_name, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            default=lambda o: make_serializable(o),
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
        )


def make_serializable(obj: Any) -> Union[int, float, List[Union[int, float]], Any]:
    """
    Converts a given object into a serializable format.

    Args:
    - obj: Any Python object

    Returns:
    - If obj is an integer or numpy integer, returns the integer value as an int
    - If obj is a numpy floating-point number, returns the floating-point value
        as a float
    - If obj is a numpy array, returns the array as a list
    - Otherwise, uses the default behavior of the json.JSONEncoder to serialize obj

    """
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return json.JSONEncoder.default(None, obj)


def contains_subdirectories(directory):
    """Check if the specified directory contains any subdirectories."""
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            return True
    return False


def get_peak_memory_usage():
    """
    Returns the peak memory usage by current cuda device (in MB) if available
    """
    if not T.cuda.is_available():
        return None

    current_device = T.cuda.current_device()
    peak_memory = T.cuda.max_memory_allocated(current_device)
    return peak_memory / (1024 * 1024)


class ResourceTracker(object):
    """
    This class serves as a context manager to track time and
    memory allocated by code executed inside it.
    """

    def __init__(self, logger, monitoring_interval):
        self.logger = logger
        self.monitor = MemoryMonitor(logger=logger, interval=monitoring_interval)

    def __enter__(self):
        self.start_time = time.time()
        if T.cuda.is_available():
            T.cuda.reset_peak_memory_stats()  # Reset CUDA memory stats
            T.cuda.empty_cache()  # Clear CUDA cache

        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.monitor.stop()
        cuda_peak = get_peak_memory_usage()
        if cuda_peak:
            self.logger.info(f"CUDA Memory allocated (peak): {cuda_peak:.2f} MB")

        elapsed_time = self.end_time - self.start_time

        self.logger.info(f"Execution time: {elapsed_time:.2f} seconds")


class MemoryMonitor:
    peak_memory = 0  # Class variable to store peak memory usage

    def __init__(self, interval=20.0, logger=print):
        self.interval = interval  # Time between executions in seconds
        self.timer = None  # Placeholder for the timer object
        self.logger = logger
        self.lock = threading.Lock()  # Lock for thread-safe updates to class variables

    def monitor_memory(self):
        process = psutil.Process(os.getpid())
        children = process.children(recursive=True)
        total_memory = process.memory_info().rss

        for child in children:
            total_memory += child.memory_info().rss

        with self.lock:
            # Check if the current memory usage is a new peak and update accordingly
            MemoryMonitor.peak_memory = max(MemoryMonitor.peak_memory, total_memory)

    def _schedule_monitor(self):
        """Internal method to execute monitor_memory and schedule the next execution"""
        self.monitor_memory()
        self.schedule_next()

    def schedule_next(self):
        """Schedules the next execution of the monitor task"""
        if not self.timer or not self.timer.is_alive():
            self.timer = threading.Timer(self.interval, self._schedule_monitor)
            self.timer.start()

    def start(self):
        """Starts the periodic monitoring"""
        if not self.timer or not self.timer.is_alive():
            threading.Thread(target=self._schedule_monitor).start()

    def stop(self):
        """Stops the periodic monitoring"""
        if self.timer and self.timer.is_alive():
            self.timer.cancel()
            self.timer = None
        self.logger.info(
            f"CPU Memory allocated (peak): {MemoryMonitor.peak_memory / (1024**2):.2f} MB"
        )

    @classmethod
    def get_peak_memory(cls):
        """Returns the peak memory usage"""
        return cls.peak_memory
