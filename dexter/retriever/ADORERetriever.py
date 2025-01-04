from typing import Dict, List, Union, Tuple, Any

from torch import Tensor
from torch.optim import AdamW
import torch
from transformers import get_scheduler

import numpy as np
from tqdm import tqdm

from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.evidence import Evidence
from dexter.retriever.dense.HfRetriever import HfRetriever
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity

similarity_metric = CosineSimilarity()

def calculate_rr(all_doc_idxs, relevant_doc_idxs_set):
    rr = 0
    for rank, doc_idx in enumerate(all_doc_idxs, start=1):
        if doc_idx in relevant_doc_idxs_set:
            rr = 1 / rank
            break
    return rr

def compute_loss_for_query_and_hard_negatives(q_idx, relevant_doc_idxs, all_hard_negative_idxs, similarity_scores):
    relevant_doc_idxs_set = set(relevant_doc_idxs)

    merged_doc_idxs = relevant_doc_idxs + all_hard_negative_idxs

    merged_doc_idxs = sorted(merged_doc_idxs, key=lambda idx: similarity_scores[q_idx, idx], reverse=True)

    idxs_to_pos = dict()

    for i in range(len(merged_doc_idxs)):
        idxs_to_pos[merged_doc_idxs[i]] = i

    loss = 0

    for rel_doc_idx in relevant_doc_idxs:
        for hard_negative_idx in all_hard_negative_idxs:
            l_r = np.log(1 + np.exp(similarity_scores[q_idx, hard_negative_idx] - similarity_scores[q_idx, rel_doc_idx]))

            orig_rr = calculate_rr(merged_doc_idxs, relevant_doc_idxs_set)

            relevant_pos = idxs_to_pos[rel_doc_idx]
            negative_pos = idxs_to_pos[hard_negative_idx]

            merged_doc_idxs[relevant_pos] = hard_negative_idx
            merged_doc_idxs[negative_pos] = rel_doc_idx

            switched_rr = calculate_rr(merged_doc_idxs, relevant_doc_idxs_set)

            loss += float((orig_rr - switched_rr) * l_r)

    return loss / (len(relevant_doc_idxs) * len(relevant_doc_idxs_set))


class ADORERetriever(HfRetriever):
    def __init__(self,config=DenseHyperParams) -> None:
        super().__init__(config)
        self.passage_embeddings: Union[List[Tensor], np.ndarray, Tensor] = None

    def train(self, queries: List[Question],
              corpus: List[Evidence],
              qrels: Dict[str, Dict[str, int]],
              top_k: int,
              n_epochs: int
              ) -> None:


        if self.passage_embeddings is None:
            print("BUILDING PASSAGE EMBEDDINGS FOR ADORE TRAINING - THIS MIGHT TAKE A WHILE")
            self.passage_embeddings = self.encode_corpus(corpus)
        else:
            print("USING EXISTING PASSAGE EMBEDDINGS FOR ADORE TRAINING")

        relevant_docs_raw_idxs = dict() # maps queries to doc positions (NOT DOC IDS)!
        for query in queries:
            relevant_docs_raw_idxs[query.id()] = []
            for raw_doc_idx, doc in enumerate(corpus):
                if qrels[query.id()][doc.id()] == 1:
                    relevant_docs_raw_idxs[query.id()].append(raw_doc_idx)

        num_training_steps = n_epochs * (len(queries) / self.batch_size)
        optimizer = AdamW(self.question_encoder.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.question_encoder.to(device)
        self.question_encoder.train()

        progress_bar = tqdm(range(num_training_steps))


        for epoch in range(n_epochs):
            for i in range(0, len(queries), self.batch_size):
                cur_queries = queries[i : i + self.batch_size]
                query_embeddings = self.encode_queries(cur_queries)


                similarity_scores = similarity_metric.evaluate(query_embeddings, self.passage_embeddings) # TENSOR SHAPE IS |Q| * |C|

                top_k_values, top_k_idxs =torch.topk(similarity_scores, min(top_k + 20, len(similarity_scores[1])), dim=1, largest=True, sorted=True)
                top_k_idxs = top_k_idxs.cpu().numpy()
                # top_k_values = top_k_values.cpu().numpy()
                batch_total_loss = 0

                for q_idx, query in enumerate(cur_queries):

                    all_hard_negative_idxs= [
                        doc_raw_idx
                        for doc_raw_idx in top_k_idxs[q_idx]
                        if corpus[doc_raw_idx].id() not in qrels[query.id()]
                    ]

                    assert len(all_hard_negative_idxs) >= top_k
                    all_hard_negative_idxs = all_hard_negative_idxs[:top_k]

                    batch_total_loss += compute_loss_for_query_and_hard_negatives(q_idx, relevant_docs_raw_idxs[query.id()], all_hard_negative_idxs, similarity_scores)

                loss = torch.Tensor(batch_total_loss / len(cur_queries), device=device)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
