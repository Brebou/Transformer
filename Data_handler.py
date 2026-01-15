import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, TensorDataset

def prepare_data():
    df = pd.read_csv('tatoeba/eng-fra.txt', sep='\t', names=('en', 'fr'))
    df['sample_fr'] = df.apply(lambda row: f"ddd {row['fr']} fff", axis=1)
    df['sample_en'] = df.apply(lambda row: f"ddd {row['en']} fff", axis=1)

    vec_en = CountVectorizer(token_pattern=r"\b\w+\b", lowercase=False)
    vec_fr = CountVectorizer(token_pattern=r"\b\w+\b", lowercase=False)
    vec_fr.fit(df['sample_fr'])
    vec_en.fit(df['sample_en'])

    analyzer_fr = vec_fr.build_analyzer()
    analyzer_en = vec_en.build_analyzer()

    df['tokens_fr'] = df['sample_fr'].map(analyzer_fr)
    df['tokens_fr'] = df['tokens_fr'].map(lambda x: list(map(vec_fr.vocabulary_.get, x)))
    df['tokens_en'] = df['sample_en'].map(analyzer_en)
    df['tokens_en'] = df['tokens_en'].map(lambda x: list(map(vec_en.vocabulary_.get, x)))
    max_len_fr = df['tokens_fr'].map(len).max()
    max_len_en = df['tokens_en'].map(len).max()

    vec_en.vocabulary_['PPPading'] = len(vec_en.vocabulary_)
    vec_fr.vocabulary_['PPPading'] = len(vec_fr.vocabulary_)
    return df, vec_en, vec_fr, max_len_en, max_len_fr

class VocabDataset(TensorDataset):
    def __init__(self, 
        dataframe,
        padding_fr,
        padding_en):
        self.dataframe = dataframe
        self.padding_fr = padding_fr
        self.padding_en = padding_en
        self.df = dataframe.reset_index(drop=True)         
        self.max_len_fr = self.df['tokens_fr'].map(len).max()
        self.max_len_en = self.df['tokens_en'].map(len).max()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        tokens_fr = self.df.loc[idx, 'tokens_fr']
        tokens_en = self.df.loc[idx, 'tokens_en']
        tokens_fr = tokens_fr + [self.padding_fr] * (self.max_len_fr - len(tokens_fr))
        tokens_en = tokens_en + [self.padding_en] * (self.max_len_en - len(tokens_en))

        return torch.tensor(tokens_en, dtype=torch.long), torch.tensor(tokens_fr, dtype=torch.long)

    