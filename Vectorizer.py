class Word2vecVectoriser(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, df, input_vocab, class_vocab, sent_embed, is_sequence, data_field, vector_len=-1, load_path=None):
        """
        Args:
            review_vocab (Vocabulary): maps words to integers
            rating_vocab (Vocabulary): maps class labels to integers
        """
        self.df = df
        self.data = df[data_field]
        self.input_vocab = input_vocab
        self.class_vocab = class_vocab
        self.is_sequence = is_sequence
        self.sent_embed = sent_embed 
        self.vector_len = vector_len
        if not load_path:
            self.model = self.model_base()
            self.build_vocab()
            self.train_model()
        else:
            self.model = gensim.models.KeyedVectors.load(load_path)
        
    def model_base(self):
        cores = 4
        model = Word2Vec(min_count=1,
                         window=2,
                         vector_size=100,
                         sample=6e-5, 
                         alpha=0.03, 
                         min_alpha=0.0007, 
                         negative=20,
                         workers=cores-1)
        return model

    def build_vocab(self):
        t = time()
        self.model.build_vocab(self.data, progress_per=10)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    def train_model(self):
        t = time()
        self.model.train(self.data, total_examples=self.model.corpus_count, epochs=50, report_delay=1)
        print('Time to train : {} mins'.format(round((time() - t) / 60, 2)))
        
    def vectorize(self, words):
        """Create a collapsed one hot vector for the review
        Args:
            review (str): the review
        Returns:
            one_hot (np.ndarray): the collapsed onehot encoding
        """

        indicies = [self.model.wv.get_index(word) for word in words]
        vectors = self.model.wv[indicies]

        if self.sent_embed:
            frequency = [self.input_vocab.lookup_frequency(word) for word in words]
            return sum([vec*freq for vec,freq in zip(vectors, frequency)])

        return vectors.flatten()
              
    @classmethod
    def from_dataframe(cls, df, is_sequence, data_field="tfidf10", feature_field="is_fulltime", sent_embed=False, cutoff=25):
        """Instantiate the vectorizer from the dataset dataframe
        Args:
            review_df (pandas.DataFrame): the review dataset
            cutoff (int): the parameter for frequency based filtering
        Returns:
            an instance of the ReviewVectorizer
        """
        class_vocab = Vocabulary(add_unk=False)
        for category in sorted(set(df[feature_field])):
            class_vocab.add_token(category)

        word_counts = Counter()
        for title in df[data_field]:
            for token in title:
                if token not in string.punctuation:
                    word_counts[token] += 1

        if is_sequence:
            input_vocab = SequenceVocabulary()
        else:
            input_vocab = Vocabulary(add_unk=True)
        # Add top words if count > provided count
        for word, count in word_counts.items():
            if count > cutoff:
                input_vocab.add_token(word, count)

        return cls(input_vocab, class_vocab, sent_embed, is_sequence)

    @property
    def num_features(self):
        if self.sent_embed:
            return 100 #word2vec model size
        return self.vector_len * 100
    
    @classmethod
    def from_serializable(cls, contents):
        """Intantiate a ReviewVectorizer from a serializable dictionary
        Args:
            contents (dict): the serializable dictionary
        Returns:
            an instance of the ReviewVectorizer class
        """
        review_vocab = Vocabulary.from_serializable(contents['input_vocab'])
        rating_vocab = Vocabulary.from_serializable(contents['class_vocab'])
        return cls(input_vocab=input_vocab, class_vocab=class_vocab)
    
    def to_serializable(self):
        """Create the serializable dictionary for caching
        Returns:
            contents (dict): the serializable dictionary
        """
        return {'input_vocab': self.input_vocab.to_serializable(),
                'class_vocab': self.class_vocab.to_serializable()}
    
