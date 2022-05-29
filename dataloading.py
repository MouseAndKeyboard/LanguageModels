import pandas as pd

def decodeList( x): return [w.strip("'") for w in x.split(' ')]

def pythonise( df):
    if 'is_fulltime' in df.keys():
        df.is_fulltime = df.is_fulltime.astype(str)
    if 'job_description' in df.keys():
        df['job_description'] = df['job_description'].apply(decodeList)
    if 'tfidf10' in df.keys():
        df['tfidf10'] = df['tfidf10'].apply(decodeList)

def load_data( name, send_computer_to_hell=True):
    nrows = None
    if not send_computer_to_hell:
        nrows = 500

    df = pd.read_csv(f'./data/{name}.csv', nrows=nrows)
    train = pd.read_csv(f'./data/{name}-train.csv', nrows=nrows)
    test = pd.read_csv(f'./data/{name}-test.csv', nrows=nrows)
    val = pd.read_csv(f'./data/{name}-val.csv', nrows=nrows)
    
    # ugly af
    for d in [df, train, test, val]:
        pythonise(d)

    return df, train, test, val
