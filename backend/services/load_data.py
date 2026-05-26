import pandas as pd

class DataLoader:

    @staticmethod
    def load(file_path):

        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)

        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)

        elif file_path.endswith(".json"):
            return pd.read_json(file_path)

        else:
            raise Exception(
                "Unsupported file"
            )