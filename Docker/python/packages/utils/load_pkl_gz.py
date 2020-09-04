import pickle, gzip


def load_pkl_gz(input_path: str, encoding: str = 'latin1'):
    with gzip.open(input_path, 'rb') as f:
        return pickle.load(f, encoding=encoding)
