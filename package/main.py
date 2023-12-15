import attr
import pandas as pd
import numpy as np
import nltk
import spacy

from datetime import datetime
from functools import cached_property
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
                if stopwords and token.text in stopwords:
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
    new_val = {metric_type: {metric_name: [] for metric_name in metric_dict.keys()}
               for metric_type, metric_dict in list_of_values_dicts[0].items()}
    for values in list_of_values_dicts:
        for metric_type, d in values.items():
            for metric_name, metric_value in d.items():
                if metric_name not in new_val[metric_type]:
                    new_val[metric_type][metric_name] = [metric_value]
                new_val[metric_type][metric_name].append(metric_value)

    return {metric_type: {metric_name: np.mean(metric_values) for metric_name, metric_values in metric_dict.items()}
            for metric_type, metric_dict in new_val.items()}


def pipe_texts(texts: List[str], config: Config) -> Iterable[Dict[str, Dict[str, float]]]:
    docs = config.nlp.pipe(texts)
    for doc in docs:
        data = TextData(doc.text, config=config, doc=doc)
        processed = ProcessTextData(data)
        yield processed.values


def values_to_flat_dict(values: Dict[str, Dict[str, float]]) -> Dict:
    d_p = {}
    for dd in values.values():
        if type(dd) == dict:
            d_p.update(dd)
    return d_p


def main():
    # some_large_text = "Der Dackel zeichnet sich durch niedrige, kurzläufige, langgestreckte, aber kompakte Gestalt " \
    #                   "aus. Er ist sehr muskulös, mit aufrechter Haltung des Kopfes und aufmerksamem " \
    #                   "Gesichtsausdruck. Die verkürzten Beine der Dackel sind das Resultat einer gezielten Selektion " \
    #                   "auf Chondrodysplasie und sind im Rassestandard verankert. Es gibt die Hunde in verschiedenen " \
    #                   "Größen und Fellvariationen: Langhaar, Rauhaar, Kurzhaar in jeweils vielen Farbvariationen, " \
    #                   "auch mehrfarbig, gestromt oder gefleckt. Während gefleckte Tiere als eine der 4 Färbungen " \
    #                   "durch den FCI definiert sind, werden Schwarze Tiere ohne Brand sowie weiße mit oder ohne Brand " \
    #                   "im Rassestandard der FCI ausdrücklich ausgeschlossen. Unter AKC-Regeln können weiß " \
    #                   "gescheckte Dackel ausgestellt werden und sind dort als Piebalds bekannt. Die hängenden " \
    #                   "Ohren sind nicht zu weit vorne angesetzt, ausreichend, aber nicht übertrieben lang und " \
    #                   "abgerundet. In den drei Haararten werden die Dackel im FCI-Standard nach ihrer Größe " \
    #                   "unterschieden in Teckel (T) (früher Normalteckel), Brustumfang (BU) über 35 cm, " \
    #                   "Gewichtsobergrenze etwa 9 kg, Zwergteckel (Zw), BU über 30 bis 35 cm, und Kaninchenteckel (" \
    #                   "Kt), BU bis 30 cm. "
    # data_de = TextData(text=some_large_text, config=Config(lang='de'))
    #
    # some_large_text_ru = "Знаете ли вы? Будапештский университет иудаики, около 1890 года Социалистическая власть не " \
    #                      "мешала работе раввинской семинарии (на илл.). Будапештский университет иудаики был открыт в " \
    #                      "1877 году, через несколько десятилетий после того, как в Падуе, Меце, Париже и Бреслау были " \
    #                      "построены первые европейские раввинские семинарии. Тем не менее, он остаётся старейшим " \
    #                      "существующим учебным заведением в мире, где готовят раввинов. "
    # data_ru = TextData(text=some_large_text_ru, config=Config(lang='ru'))
    #
    # dd = ProcessTextData(data_de)
    # print(dd.values)
    #
    # dr = ProcessTextData(data_ru)
    # print(dr.values)

    start = datetime.now()
    config_de = Config(lang='de')
    df = pd.read_csv('/Users/galina.ryazanskaya/Downloads/thesis?/code?/split_questions_cp_0.tsv', sep='\t',
                     index_col=0).dropna()
    df.reset_index(inplace=True)
    long = df.melt(id_vars=['index'], value_name='transcript', var_name='task').sort_values(['index', 'task'])
    texts = long['transcript'].to_list()
    print('time for read and melt:', datetime.now() - start)

    start = datetime.now()
    vals = list(pipe_texts(texts, config_de))
    print('time for pipe:', datetime.now() - start)
    long['values'] = vals

    start = datetime.now()
    for_pd = []
    for_entire_pd = []
    cur_vals = ['', []]
    for i, (ix, row) in enumerate(long.iterrows()):
        if row['task'] == 'preprocessed_transcript':
            dp = values_to_flat_dict(row['values'])
            for_entire_pd.append(dp)
        else:
            if i > 0 and row['index'] != long.iloc[i-1]['index']:
                vals = average_values(cur_vals[-1])
                vals = values_to_flat_dict(vals)
                vals['index'] = cur_vals[0]
                for_pd.append(vals)
                cur_vals = ['', []]
            else:
                cur_vals[0] = row['index']
                cur_vals[-1].append(row['values'])
    new_df = pd.DataFrame(for_pd)
    print('time for pd:', datetime.now() - start)

    # values_df = pd.pivot(long, columns='task', index='index')['values']
    # print(values_df.head())
    #
    # averages = values_df[['anger', 'sadness', 'fear', 'happiness']].apply(average_values, axis=1)
    # print(averages)

    start = datetime.now()
    newrows = {}
    for index, row in df[['anger', 'sadness', 'fear', 'happiness']].iterrows():
        vals = []
        for w in row[:-1]:
            v = ProcessTextData(TextData(text=w, config=config_de)).values
            vals.append(v)
        newrows[index] = average_values(vals)

    for_pd = []
    for i, d in newrows.items():
        d_p = {'index': i}
        for dd in d.values():
            d_p.update(dd)
        for_pd.append(d_p)
    new_df = pd.DataFrame(for_pd)

    with open('NET_NAP_new_graph_limit100.tsv', 'w') as f:
        f.write(new_df.to_csv(sep='\t'))

    new_rows_entire = {}
    for index, w in df['preprocessed_transcript'].items():
        new_rows_entire[index] = ProcessTextData(TextData(text=w, config=config_de)).values

    for_pd = []
    for i, d in new_rows_entire.items():
        d_p = {'index': i}
        for dd in d.values():
            d_p.update(dd)
        for_pd.append(d_p)
    new_df = pd.DataFrame(for_pd)

    with open('NET_NAP_new_graph_limit100_entire.tsv', 'w') as f:
        f.write(new_df.to_csv(sep='\t'))

    print('time for non-pipe + pd:', datetime.now() - start)


if __name__ == '__main__':
    main()
