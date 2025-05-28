import pandas as pd
import os


def load_and_clean_normal_dataset(log_file_path: str) -> pd.DataFrame:
    '''
    Loads a log file of a normal dataset, converts data types, and performs
    comprehensive cleaning.

    :param log_file_path: The full path to the CSV log file.
                          e.g., 'WinLogAno/data/raw/evtx_logs/WindowsEventLogs_Last30Days.csv'
    :return: A cleaned and processed pandas DataFrame.
    '''
    # --- 1. File Loading and Validation ---
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"The log file {log_file_path} does not exist.")

    try:
        print(f"Loading log file from: {log_file_path}")
        df = pd.read_csv(log_file_path)

        # --- 2. Initial Data Type Conversion ---
        print("Converting data types...")
        # Convert the 'TimeCreated' column to datetime format. Coerce will turn failures into NaT (Not a Time).
        df['TimeCreated'] = pd.to_datetime(df['TimeCreated'], errors='coerce', utc=True)
        # Convert 'Id' column to integer. Coerce will turn failures into NaN.
        df['Id'] = pd.to_numeric(df['Id'], errors='coerce')

        # --- 3. Data Cleaning and Standardization ---
        print("Cleaning and standardizing data...")

        # Drop rows where essential columns failed to parse, as they are unusable.
        df.dropna(subset=['TimeCreated', 'Id'], inplace=True)

        # Convert 'Id' to integer type after dropping NaNs.
        df['Id'] = df['Id'].astype(int)

        # Standardize column names to be more script-friendly (lowercase and underscores)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Clean string columns: remove leading/trailing whitespace and fill any remaining NaN.
        for col in ['leveldisplayname', 'providername', 'message', 'logname']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().fillna('N/A')


        # Sort the DataFrame by time, which is critical for any sequential or time-series analysis.
        df.sort_values(by='timecreated', inplace=True)

        print("Data loading and cleaning complete.")


        # --- 4. Final DataFrame Structure ---
        print("Final DataFrame structure:")
        print(df.info())

        return df

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        # Return an empty DataFrame in case of a critical error.
        return pd.DataFrame()


