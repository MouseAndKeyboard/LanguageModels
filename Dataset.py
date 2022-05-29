
import torch
from torch.utils.data import Dataset, DataLoader
from dataloading import *
from Vocabulary import *
import numpy as np



class JobDescriptionDataset(Dataset):
    def __init__(self, df, train, val, test, vectorizer, data_field="tfidf10", feature_field="is_fulltime"):
        """
        Args:
        review_df (pandas.DataFrame): the dataset
        vectorizer (Vectorizer): vectorizer instantiated from dataset
        """
        self.df = df
        self._vectorizer = vectorizer
        self.feature_field = feature_field
        self.data_field = data_field

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        def measure_len(context): return len(context)
        self._max_seq_length = max(map(measure_len, df[self.data_field])) + 2

        # if we don't have constant length 
        if not all(len(val) == self._max_seq_length for val in self.data_field):
            self.vector_len = self._max_seq_length
        else:
            self.vector_len = -1
        
        self.train = train
        self.val = val
        self.test = test
        self.train_size = len(self.train)
        self.validation_size = len(self.val)
        self.test_size = len(self.test)
        self._lookup_dict = {'train': (self.train, self.train_size),
                             'val': (self.val, self.validation_size),
                             'test': (self.test, self.test_size)}
        self.set_split('train')

        # Class weights
        class_counts = df[self.feature_field].value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.class_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / \
            torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, name, vectoriser, 
                                         data_field, 
                                         feature_field,
                                         is_sequence,
                                         sent_embed,
                                         cutoff=25):
        """Load dataset and make a new vectorizer from scratch
        Args:
             name (str): name of task to load
        Returns:
            an instance of ReviewDataset
        """
        df, train, test, val = load_data(name)
        return cls(df, train, test, val, vectoriser.from_dataframe(df, is_sequence, data_field=data_field, 
            feature_field=feature_field, sent_embed=sent_embed, cutoff=cutoff), data_field=data_field, feature_field=feature_field)

    @classmethod
    def load_dataset_and_load_vectorizer(cls, name, vectorizer_filepath):
        """Load dataset and the corresponding vectorizer. 
        Used in the case in the vectorizer has been cached for re-use

        Args:
            news_csv (str): location of the dataset
            vectorizer_filepath (str): location of the saved vectorizer
        Returns:
            an instance of NewsDataset
        """
        df, train, test, val = load_data(name)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, train, test, val, vectorizer.from_dataframe(df))

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """a static method for loading the vectorizer from file

        Args:
            vectorizer_filepath (str): the location of the serialized vectorizer
        Returns:
            an instance of NewsVectorizer
        """
        with open(vectorizer_filepath) as fp:
            return NewsVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """saves the vectorizer to disk using json

        Args:
            vectorizer_filepath (str): the location to save the vectorizer
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe
        Args:
        split (str): one of "train", "val", or "test"
        """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        Args:
        index (int): the index to the data point
        Returns:
        a dict of the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        input_vector = \
            self._vectorizer.vectorize(row[self.data_field], vector_length=self.vector_len)
        class_index = \
            self._vectorizer.class_vocab.lookup_token(row[self.feature_field])
        return {'x_data': input_vector,
                'y_target': class_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        Args:
        batch_size (int)
        Returns:
        number of batches in the dataset
        """
        return len(self) // batch_size



