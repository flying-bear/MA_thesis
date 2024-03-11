import attr
import pandas as pd
import numpy as np
import nltk
import spacy

from datetime import datetime
from functools import cached_property

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertForNextSentencePrediction
from typing import Iterable
from wordfreq import word_frequency


from graph import *
from lms import *
from lex import *
from synt import *

# global
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

nlp_de = spacy.load("de_core_news_md")
nlp_ru = spacy.load("ru_core_news_md")

stopwords_de = nltk.corpus.stopwords.words("german")
stopwords_ru = nltk.corpus.stopwords.words("russian")

model_name_de = "bert-base-german-cased"
model_de = AutoModel.from_pretrained(model_name_de)
tokenizer_de = AutoTokenizer.from_pretrained(model_name_de)
tokenizer_nsp_de = BertTokenizer.from_pretrained(model_name_de)
model_nsp_de = BertForNextSentencePrediction.from_pretrained(model_name_de)

model_name_ru = "DeepPavlov/rubert-base-cased"
model_ru = AutoModel.from_pretrained(model_name_ru)
tokenizer_ru = AutoTokenizer.from_pretrained(model_name_ru)
tokenizer_nsp_ru = BertTokenizer.from_pretrained(model_name_ru)
model_nsp_ru = BertForNextSentencePrediction.from_pretrained(model_name_ru)


def flatten(list_of_lists: List[List]) -> List:
    return [item for sublist in list_of_lists for item in sublist]


def idf_sent_vectors(words: List[str], vectors: List[np.array], lang='de') -> np.array:
    assert len(words) > 0
    assert len(words) == len(vectors)
    weights = [word_frequency(w, lang=lang) for w in words]
    if sum(weights) == 0:  # all words are OOV for wordfreq
        return np.average(vectors, axis=0)
    return np.average(vectors, axis=0, weights=weights)


@attr.s(auto_attribs=True)
class Config:
    lang: Optional[str] = attr.ib(default='de')
    stopwords: Optional[List[str]] = attr.ib(default=None)

    def __attrs_post_init__(self):
        if self.lang not in ['de', 'ru']:
            raise ValueError(f'lang={self.lang} is not supported')
        if self.stopwords is None:
            self.stopwords = stopwords_de if self.lang == 'de' else stopwords_ru
        if self.lang == 'de':
            self.nlp = nlp_de
            self.bert_model = model_de
            self.bert_model_nsp = model_nsp_de
            self.bert_tokenizer = tokenizer_de
            self.bert_tokenizer_nsp = tokenizer_nsp_de
        elif self.lang == 'ru':
            self.nlp = nlp_ru
            self.bert_model = model_ru
            self.bert_model_nsp = model_nsp_ru
            self.bert_tokenizer = tokenizer_ru
            self.bert_tokenizer_nsp = tokenizer_nsp_ru


@attr.s(auto_attribs=True)
class TextData:
    text: str
    config: Config
    doc: Optional[spacy.tokens.doc.Doc] = attr.ib(default=None)
    words: Optional[List[str]] = attr.ib(factory=list)
    lemmas: Optional[List[str]] = attr.ib(factory=list)
    pos: Optional[List[str]] = attr.ib(factory=list)
    sents: Optional[List[str]] = attr.ib(factory=list)
    sent_words: Optional[List[List[str]]] = attr.ib(factory=list)

    word_vectors: Optional[List[np.array]] = attr.ib(factory=list)
    word_for_vectors: Optional[List[str]] = attr.ib(factory=list)

    sent_word_vectors: Optional[List[List[np.array]]] = attr.ib(factory=list)
    sent_word_for_vectors: Optional[List[List[str]]] = attr.ib(factory=list)
    oov: Optional[List[str]] = attr.ib(factory=list)

    def __attrs_post_init__(self):
        self.build_with_spacy(self.text)
        self.sent_vectors = [idf_sent_vectors(words, vectors, lang=self.config.lang) for
                             words, vectors in zip(self.sent_word_for_vectors, self.sent_word_vectors)]

    def build_with_spacy(self, text: str):
        stopwords = [] if self.config.stopwords is None else self.config.stopwords
        outer_doc = self.config.nlp(text) if self.doc is None else self.doc

        for sent in outer_doc.sents:
            self.sents.append(sent.text)
            sent_tokens = []
            sent_vectors = []
            sent_word_for_vectors = []
            for token in sent:
                if stopwords and token.text.lower() in stopwords:
                    continue
                self.words.append(token.text)
                self.lemmas.append(token.lemma_)
                self.pos.append(token.pos_)
                if token.pos_ not in ['PUNCT', 'NUM']:
                    # filter out unvectorizable tokens
                    if token.is_oov:
                        self.oov.append(token.text)
                    else:
                        sent_vectors.append(token.vector)
                        sent_word_for_vectors.append(token.text)
                        self.word_vectors.append(token.vector)
                        self.word_for_vectors.append(token.text)
                sent_tokens.append(token.text)
            if sent_vectors:
                # filter out empty sentences
                self.sent_word_vectors.append(sent_vectors)
                self.sent_word_for_vectors.append(sent_word_for_vectors)
            self.sent_words.append(sent_tokens)


@attr.s(auto_attribs=True)
class ProcessTextData:
    data: TextData

    # lexical
    @cached_property
    def n_words(self):
        return len(self.data.words)

    @cached_property
    def LTR(self):
        return self.TTR()

    @cached_property
    def MALTR(self):
        return self.MATTR()

    @cached_property
    def lexical_features(self):
        return {'n_words': self.n_words, 'LTR': self.LTR, 'MALTR': self.MALTR}

    # syntactic
    @cached_property
    def sentence_stats(self) -> Dict[str, float]:
        sent_lens = [len(s) for s in self.data.sent_words]
        return {'mean_sent_len': np.mean(sent_lens),
                'std_sent_len': np.std(sent_lens),
                'min_sent_len': np.min(sent_lens),
                'max_sent_len': np.max(sent_lens)}

    @cached_property
    def n_sents(self) -> int:
        return len(self.data.sents)

    @cached_property
    def mean_sent_words(self) -> float:
        return self.sentence_stats['mean_sent_len']

    @cached_property
    def std_sent_words(self) -> float:
        return self.sentence_stats['std_sent_len']

    @cached_property
    def min_sent_words(self) -> int:
        return self.sentence_stats['min_sent_len']

    @cached_property
    def max_sent_words(self) -> int:
        return self.sentence_stats['max_sent_len']

    @cached_property
    def pos_rates(self) -> Dict[str, float]:
        return pos_rates(self.data.pos)

    @cached_property
    def syntactic_features(self) -> Dict[str, float]:
        syntactic = {'n_sents': self.n_sents}
        syntactic.update(self.sentence_stats)
        syntactic.update(self.pos_rates)
        return syntactic

    # graph
    @cached_property
    def graph_features(self) -> Dict[str, float]:
        return self.graph_statistics()

    # LM
    @cached_property
    def mean_lcoh(self) -> float:
        return np.mean(self.local_coherence_list())

    @cached_property
    def mean_gcoh(self) -> float:
        return np.mean(self.global_coherence_list())

    @cached_property
    def mean_cgcoh(self) -> float:
        return np.mean(self.cumulative_global_coherence_list())

    @cached_property
    def mean_scoh(self) -> float:
        return np.mean(self.second_order_coherence_list())

    @cached_property
    def mean_sent_prob(self) -> float:
        return np.mean(self.sent_prob_list)

    @cached_property
    def LM_features(self) -> dict[str, float]:
        return {'m_lcoh': self.mean_lcoh,
                'm_gcoh': self.mean_gcoh,
                'm_cgcoh': self.mean_cgcoh,
                'm_scoh': self.mean_scoh,
                'm_sporb': self.mean_sent_prob,
                'm_bert_lcoh': np.mean(self.local_coherence_list(model='bert')),
                'm_bert_gcoh': np.mean(self.global_coherence_list(model='bert')),
                'm_bert_cgcoh': np.mean(self.cumulative_global_coherence_list(model='bert')),
                'm_bert_scoh': np.mean(self.second_order_coherence_list(model='bert'))
                }

    # lexical
    def TTR(self, lemmas: bool = True) -> float:
        items = self.data.lemmas if lemmas else self.data.words
        return TTR(items)

    def MATTR(self, w: Optional[int] = 10, lemmas: bool = True) -> float:
        items = self.data.lemmas if lemmas else self.data.words
        return MATTR(items, w=w)

    # graph
    def graph_statistics(self, w: Optional[int] = 100, lemmas: bool = True) -> Dict[str, float]:
        items = self.data.lemmas if lemmas else self.data.words
        return moving_graph_statistics(items, w=w)

    # LM
    @cached_property
    def bert_sent_vectors(self) -> List[np.array]:
        return [bert_sent_vectorize(s,
                                    model=self.data.config.bert_model,
                                    tokenizer=self.data.config.bert_tokenizer)
                for s in self.data.sents]

    def sent_vectors(self, model: Optional[str] = 'default') -> List[np.array]:
        if model == 'bert':
            return self.bert_sent_vectors
        else:
            return self.data.sent_vectors

    def local_coherence_list(self, model: Optional[str] = 'default') -> List[float]:
        return get_local_coherence_list(self.sent_vectors(model=model))

    def global_coherence_list(self, model: Optional[str] = 'default') -> List[float]:
        return get_global_coherence_list(self.sent_vectors(model=model))

    def cumulative_global_coherence_list(self, model: Optional[str] = 'default') -> List[float]:
        return get_cumulative_global_coherence_list(self.sent_vectors(model=model))

    def second_order_coherence_list(self, model: Optional[str] = 'default') -> List[float]:
        return get_second_order_coherence_list(self.sent_vectors(model=model))

    @cached_property
    def sent_prob_list(self) -> List[float]:
        return get_prob_list(self.data.sents,
                             model_nsp=self.data.config.bert_model_nsp,
                             tokenizer_nsp=self.data.config.bert_tokenizer_nsp)

    # all values
    @cached_property
    def values(self) -> Dict[str, Dict[str, float]]:
        return {'LM': self.LM_features, 'syntactic': self.syntactic_features,
                'lexical': self.lexical_features, 'graph': self.graph_features}


def average_values(list_of_values_dicts: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    if not list_of_values_dicts:
        return {}
    new_val = {metric_type: {metric_name: [] for metric_name in metric_dict.keys()}
               for metric_type, metric_dict in list_of_values_dicts[0].items()}
    for values in list_of_values_dicts:
        for metric_type, d in values.items():
            for metric_name, metric_value in d.items():
                if metric_name not in new_val[metric_type]:
                    new_val[metric_type][metric_name] = [metric_value]
                new_val[metric_type][metric_name].append(metric_value)

    return {metric_type: {metric_name: np.nanmean(metric_values) for metric_name, metric_values in metric_dict.items()}
            for metric_type, metric_dict in new_val.items()}


def pipe_texts(texts: List[str], config: Config) -> Iterable[Dict[str, Dict[str, float]]]:
    docs = config.nlp.pipe(texts)
    for doc in docs:
        data = TextData(doc.text, config=config, doc=doc)
        processed = ProcessTextData(data)
        yield processed.values


def values_to_flat_dict(values: Dict[str, Dict[str, float]], index: Optional[str] = None) -> Dict:
    # TODO: replace with something like pd.DataFrame.from_dict(ex_1, orient='index').stack().to_frame()
    d_p = {}
    for dd in values.values():
        if type(dd) == dict:
            d_p.update(dd)
    if index:
        d_p['index'] = index
    return d_p


def dict_to_layered_pd(d: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([pd.DataFrame.from_dict(d, orient='index').stack()])


def create_layered_df_for_nan() -> pd.DataFrame:
    empty_df = pd.DataFrame([np.nan])
    empty_df.columns = pd.MultiIndex.from_tuples([('type', 'metric')])
    return empty_df


def process_dataframe(df: pd.DataFrame, config: Config, not_average: Optional[List[str]] = None) -> pd.DataFrame:
    start = datetime.now()
    for_pd = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        task_values = {}
        to_average = []
        for k, w in row.items():
            if pd.isnull(w):
                v_df = create_layered_df_for_nan()
            else:
                d = ProcessTextData(TextData(text=w, config=config))
                v = d.values
                if not_average and k not in not_average:
                    to_average.append(v)
                v_df = dict_to_layered_pd(v)
            task_values[k] = v_df
        task_values[' averaged'] = dict_to_layered_pd(average_values(to_average))
        task_df = pd.concat(task_values.values(), keys=task_values.keys(), names=["TASK"], axis=1)
        for_pd.append(task_df)
    new_df = pd.concat(for_pd, axis=0)
    new_df.index = df.index
    new_df.dropna(axis=1, how='all', inplace=True)
    new_df.sort_index(axis=1, level=[0, 1], inplace=True)
    new_df = new_df.rename(columns={' averaged': 'averaged'})
    print('total time:', datetime.now() - start)
    return new_df


def main():
    pth = '/Users/galina.ryazanskaya/Downloads/thesis?/code?/'
    df = pd.read_csv(pth + 'rus_transcript_lex_by_task_with_dots.tsv', sep='\t', index_col=0)
    config = Config(lang='ru', stopwords=[])
    new_df = process_dataframe(df, config)  #, not_average=['preprocessed_transcript'])

    with open(pth + 'processed_values/ru_both.tsv', 'w') as f:
        f.write(new_df.to_csv(sep='\t'))

    df = pd.read_csv(pth + 'de_split_questions_cp_HC.tsv',sep='\t', index_col=0)
    config = Config(lang='de', stopwords=[])
    new_df = process_dataframe(df, config, not_average=['preprocessed_transcript'])

    with open(pth + 'processed_values/de_HC.tsv', 'w') as f:
        f.write(new_df.to_csv(sep='\t'))

    df = pd.read_csv(pth + 'de_split_questions_cp_0.tsv', sep='\t', index_col=0)
    config = Config(lang='de', stopwords=[])
    new_df = process_dataframe(df, config, not_average=['preprocessed_transcript'])

    with open(pth + 'processed_values/de_patients.tsv', 'w') as f:
        f.write(new_df.to_csv(sep='\t'))


if __name__ == '__main__':
    main()
