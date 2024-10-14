import os
import pickle
from collections import defaultdict


class TaskDBManager:
    def __init__(self, db_root):
        self.db_root = db_root
        self.task_progress_states_mapping = defaultdict(dict)
        self.error_recoding = defaultdict(dict)
        self._load_all_data()

    def _load_all_data(self):
        """
        Load all dictionaries from subdirectories under 'stata_db' and combine them into one for faster retrieval.
        """
        for subdir in os.listdir(self.db_root):
            subdir_path = os.path.join(self.db_root, subdir)
            if os.path.isdir(subdir_path):
                # Load task_progress_states_mapping.pkl
                progress_file = os.path.join(subdir_path, "task_progress_states_mapping.pkl")
                error_file = os.path.join(subdir_path, "error_recoding.pkl")

                if os.path.exists(progress_file):
                    with open(progress_file, 'rb') as f:
                        progress_data = pickle.load(f)
                        self.task_progress_states_mapping.update(progress_data)

                if os.path.exists(error_file):
                    with open(error_file, 'rb') as f:
                        error_data = pickle.load(f)
                        self.error_recoding.update(error_data)

    def find_nearest_float(self, task_name, float_val, dictionary):
        """
        Find the nearest float for the given task_name in the provided dictionary.
        """
        keys = [(t, f) for (t, f) in dictionary if t == task_name]
        if not keys:
            raise KeyError(f"No data found for task name {task_name}")
        nearest_key = min(keys, key=lambda x: abs(x[1] - float_val))
        return dictionary[nearest_key]

    def retrieve_data(self, task_name, task_float):
        """
        Retrieve the corresponding dictionary value from task_progress_states_mapping or error_recoding.
        """
        key = (task_name, task_float)
        # First, try to find the key in task_progress_states_mapping
        if key in self.task_progress_states_mapping:
            return self.task_progress_states_mapping[key]

        # If not found in task_progress_states_mapping, look in error_recoding
        if key in self.error_recoding:
            # If there's a matching task_name in task_progress_states_mapping, find the nearest float
            if any(k[0] == task_name for k in self.task_progress_states_mapping):
                return self.find_nearest_float(task_name, task_float, self.task_progress_states_mapping)
            else:
                # If no matching task_name in task_progress_states_mapping, return the error_recoding value
                return self.error_recoding[key]

        # If the key is not found in either dictionary
        raise KeyError(f"Task {task_name} with float {task_float} not found in any database")


if __name__ == '__main__':
    db_manager = TaskDBManager("state_db")
    result = db_manager.retrieve_data("turn on the sink faucet", 0.9)
    print(result)

    result = db_manager.retrieve_data("turn on the sink faucet", 0.92)
    print(result)

    result = db_manager.retrieve_data("turn on the sink faucet A", 0.8)
    print(result)
