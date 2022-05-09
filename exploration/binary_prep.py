import pandas as pd
import sys
import functools
from time import time

def get_summary(pipeline_func):
    @functools.wraps(pipeline_func)
    def wrapper(*args, **kwargs):
        assert isinstance(args[0], pd.core.frame.DataFrame) 
        result_df = pipeline_func(*args, **kwargs)
        assert isinstance(result_df, pd.core.frame.DataFrame)
        print(result_df.head(2))
        
        return result_df
         
    return wrapper

def get_data(filepath):
    print(f"reading from {filepath}")
    try:
        df = pd.read_csv(filepath)
        return df
    except:
        print("couldn't read data")
        return pd.DataFrame()

def pipeline(df, fns, name):
    intermediate = df
    print(f"RUNNING {name}")
    for i, fn in enumerate(fns):
        print(f"STEP {i+1}: {str(fn)}")
        intermediate = fn(intermediate)
    print("[DONE] ------------\n")
    return intermediate


@get_summary
def initial(df):
    return df

@get_summary
def drop_columns(df):
    result = df[["category", "job_description", "is_fulltime"]]
    return result

@get_summary
def create_binary_job(df):
    df["is_fulltime"] = df.job_type == 'Full Time'
    return df

@get_summary
def task1_fields(df):
    return df[["job_description", "is_fulltime"]]

@get_summary
def task2_fields(df):
    return df[["job_description", "category"]]

def display_balance(df):
    fulltime_prop = sum(df.is_fulltime) / len(df)
    print(f"FULL TIME %: {fulltime_prop * 100}%")
    print(f"OTHER %: {(1 - fulltime_prop) * 100}%")
    return df


def main():
    df = pd.DataFrame()
    if len(sys.argv) == 1:
        df = get_data('../data/seek_australia.csv')
    else:
        df = get_data(sys.argv[1])

    if df.empty:
        print('was unable to load data')
        return

    result = pipeline(df, [initial, create_binary_job, drop_columns], "Initial pipeline")
    task1_result = pipeline(result, [task1_fields, display_balance], "Task 1 pipeline")
    task2_result = pipeline(result, [task2_fields], "Task 2 pipeline")

if __name__ == '__main__':
    main()
