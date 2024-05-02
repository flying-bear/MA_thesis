import torch
import numpy as np

from typing import List


def cos_sim(v1, v2):
    return np.inner(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))


def get_local_coherence_list(clause_vectors: List[np.array]) -> List[float]:
    if len(clause_vectors) <= 1:
        return [np.nan]
    local_coherence_list = []
    for i in range(len(clause_vectors)-1):
        local_coherence_list.append(cos_sim(clause_vectors[i], clause_vectors[i+1]))
    return local_coherence_list


def get_second_order_coherence_list(clause_vectors: List[np.array]) -> List[float]:
    if len(clause_vectors) <= 1:
        return [np.nan]
    SOC_list = []
    for i in range(len(clause_vectors)-2):
        SOC_list.append(cos_sim(clause_vectors[i], clause_vectors[i+2]))
    return SOC_list


def compare_text_to_a_standard_vector(clause_vectors: List[np.array], standard_vector: np.array) -> float:
    average_file_vector = np.average(clause_vectors, axis=0)
    return cos_sim(average_file_vector, standard_vector)


def get_global_coherence_list(clause_vectors: List[np.array]) -> List[float]:
    if len(clause_vectors) <= 1:
        return [np.nan]
    standard_vector = np.average(clause_vectors, axis=0)
    return [cos_sim(vec, standard_vector) for vec in clause_vectors]


def get_cumulative_global_coherence_list(clause_vectors: List[np.array]) -> List[float]:
    if len(clause_vectors) <= 1:
        return [np.nan]
    cumulative_global_coherence_list = []
    for i in range(len(clause_vectors)):
        cumulative_vector = np.average(clause_vectors[:i+1], axis=0)
        cumulative_global_coherence_list.append(cos_sim(clause_vectors[i], cumulative_vector))
    return cumulative_global_coherence_list


# BERT

def bert_sent_vectorize(text: str, model, tokenizer) -> np.array:
    inputs = tokenizer(text, return_tensors="pt")
    word_vectors, sent_vector = model(**inputs).to_tuple()
    return sent_vector.detach().squeeze().numpy()


def next_sent_prob(sent_text_1: str, sent_text_2: str, tokenizer_nsp, model_nsp) -> float:
    tokenized = tokenizer_nsp(sent_text_1, sent_text_2, return_tensors='pt')
    predict = model_nsp(**tokenized)
    pred = torch.nn.functional.softmax(predict.logits[0], dim=0)[0].item()
    return pred


def get_prob_list(sents: List[str], tokenizer_nsp, model_nsp) -> List[float]:
    return [next_sent_prob(sents[i], sents[i+1], tokenizer_nsp=tokenizer_nsp, model_nsp=model_nsp)
            for i in range(len(sents)-1)]


def pseudo_ppl_score(sentence, model_mlm, tokenizer_mlm) -> float:
    tokenize_input = tokenizer_mlm.tokenize(sentence)
    tokenize_input = ["[CLS]"]+tokenize_input+["[SEP]"]
    tensor_input = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss = model_mlm(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())


def get_pppl_list(sents: List[str], tokenizer_mlm, model_mlm) -> List[float]:
    return [pseudo_ppl_score(sent, tokenizer_mlm=tokenizer_mlm, model_mlm=model_mlm) for sent in sents]
