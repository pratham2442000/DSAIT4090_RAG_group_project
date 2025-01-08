from typing import Dict, List, Union, Tuple, Any

import os
import joblib
import math
import random

from torch import Tensor
from torch.optim import AdamW
import torch
from transformers import get_scheduler, AutoModel

import numpy as np
from tqdm import tqdm

from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.evidence import Evidence
from dexter.retriever.dense.HfRetriever import HfRetriever
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity, DotScore

RANKING_CUTOFF = 10
similarity_metric = CosineSimilarity()
ranks = torch.arange(1, RANKING_CUTOFF + 1, dtype=torch.float, device='cuda' if torch.cuda.is_available() else 'cpu')

def calculate_ap_10(top_k_idxs_1d: Tensor, relevant_doc_idxs: Tensor) -> Tensor:

    top_10_idxs = top_k_idxs_1d[:RANKING_CUTOFF]
    relevant_mask = torch.isin(top_10_idxs, relevant_doc_idxs).float()
    cum_relevance = torch.cumsum(relevant_mask, dim=0)
    precisions = cum_relevance / ranks
    precisions_for_relevant_docs = precisions * relevant_mask

    # Avoid division by zero
    relevant_sum = relevant_mask.sum()
    if relevant_sum == 0:
        # Retains gradients
        return precisions_for_relevant_docs.sum() * 0

    # If you get a division by 0 error here late in the training, ask Henry
    return precisions_for_relevant_docs.sum() / relevant_mask.sum()

def compute_loss_for_query_and_hard_negatives(relevant_doc_idxs: Tensor, all_hard_negative_idxs: Tensor, similarity_scores_1d: Tensor, top_k_idxs_1d: Tensor):

    loss = 0

    for rel_doc_idx in relevant_doc_idxs:
        for hard_negative_idx in all_hard_negative_idxs:
            l_r = torch.log(1 + torch.exp(similarity_scores_1d[hard_negative_idx] - similarity_scores_1d[rel_doc_idx]))

            orig_ap_10 = calculate_ap_10(
                top_k_idxs_1d=top_k_idxs_1d,
                relevant_doc_idxs=relevant_doc_idxs
            )

            temp_switched_idxs = top_k_idxs_1d.clone()
            temp_switched_idxs.requires_grad = False

            relevant_pos = (temp_switched_idxs == rel_doc_idx).nonzero(as_tuple=True)[0]
            negative_pos = (temp_switched_idxs == hard_negative_idx).nonzero(as_tuple=True)[0]

            temp_switched_idxs[relevant_pos] = hard_negative_idx
            temp_switched_idxs[negative_pos] = rel_doc_idx

            switched_ap_10 = calculate_ap_10(
                top_k_idxs_1d=temp_switched_idxs,
                relevant_doc_idxs=relevant_doc_idxs
            )

            loss += ((switched_ap_10 - orig_ap_10) * l_r)

    return loss / (len(relevant_doc_idxs) * len(all_hard_negative_idxs))


class ADORERetriever(HfRetriever):
    def __init__(self,config: DenseHyperParams) -> None:
        super().__init__(config)
        self.passage_embeddings: Union[List[Tensor], np.ndarray, Tensor] = None

    def encode_corpus_for_training(self, corpus: List[Evidence]) -> None:
        """
        Encodes the corpus and persists it as 2 separate files, one time as a tensor and one time as a npy array.
        TRAIN() WILL CALL THIS FUNCTION
        IF THE CORPUS ALREADY EXISTS AS AN INDEX IT WILL USE THAT
        Args:
            corpus: The corpus to encode.

        Returns: None
        """

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.passage_embeddings is None:
            print("BUILDING PASSAGE EMBEDDINGS FOR ADORE TRAINING - THIS MIGHT TAKE A WHILE")
            directory = "indices/adore/corpus"
            filename = os.path.join(directory, "index.joblib")
            npy_filename = os.path.join(directory, "npy_index.npy")

            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(filename):
                print("Loaded existing passage index")
                self.passage_embeddings = torch.tensor(joblib.load(filename), device=device)
            else:
                self.passage_embeddings = self.encode_corpus(corpus)
                joblib.dump(self.passage_embeddings, filename)
                np.save(npy_filename, self.passage_embeddings.cpu().numpy())

        print("DONE BUILDING CORPUS")

    def train(self, queries: List[Question],
              corpus: List[Evidence],
              qrels: Dict[str, Dict[str, int]],
              top_k: int,
              n_epochs: int
              ) -> None:
        """
        Begins the training process for the ADORE retriever. It will automatically call the encoder_corpus method
        but will use the pre-existing corpus index if it is present. Additionally one can choose just to encode the corpus
        by calling only that method. See the documentation above.
        Args:
            queries: The list of queries
            corpus: The list of Evidences objects
            qrels: The relationships between questions and evidence.
            top_k: how many passages we should consider at each retrieval, gets the top k
            n_epochs: Number of epochs to train the model
        Returns: None
        """

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Encode the corpus if necessary
        self.encode_corpus_for_training(corpus)

        # sanity check
        if not isinstance(self.passage_embeddings, torch.Tensor):
            self.passage_embeddings = torch.tensor(self.passage_embeddings, device=device)
        else:
            self.passage_embeddings = self.passage_embeddings.to(device)

        qrels_tensor_dict = dict()

        # Pre compute locations
        for query in queries:
            qrels_tensor_dict[query.id()] = torch.tensor(
                [int(x) for x in qrels[query.id()].keys()],
                device=device,
                dtype=torch.int64
            )

        num_training_steps = n_epochs * int(math.ceil(len(queries) / self.batch_size))
        optimizer = AdamW(self.question_encoder.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.question_encoder.to(device)
        self.question_encoder.train()

        progress_bar = tqdm(range(num_training_steps), "Epoch progress")

        for epoch in range(n_epochs):
            # poor man's data loader, we shuffle at each epoch, we can not do this using tensors, query id is a string
            random.shuffle(queries)

            for i in range(0, len(queries), self.batch_size):
                cur_queries = queries[i : i + self.batch_size]
                query_embeddings = self.encode_queries(cur_queries)

                # TENSOR SHAPE IS |Q| (batch_size) * |C| (whole corpus size)
                similarity_scores = similarity_metric.evaluate(query_embeddings, self.passage_embeddings)

                # We take +20 since relevant docs might be in the topk. We always want the topk to be hard negatives only.
                top_k_values, top_k_idxs =torch.topk(similarity_scores, min(top_k + 20, len(similarity_scores[1])), dim=1, largest=True, sorted=True)
                batch_total_loss = 0

                for q_idx, query in enumerate(cur_queries):
                    # For each query in the batch, retrieve the top-k hard negatives
                    hard_negative_mask = ~torch.isin(top_k_idxs[q_idx], qrels_tensor_dict[query.id()])
                    all_hard_negative_idxs = top_k_idxs[q_idx][hard_negative_mask]

                    assert len(all_hard_negative_idxs) >= top_k

                    # we take the top_k and eliminate the +20 part, this will only have hard-negatives
                    all_hard_negative_idxs = all_hard_negative_idxs[:top_k]

                    batch_total_loss += compute_loss_for_query_and_hard_negatives(
                        relevant_doc_idxs=qrels_tensor_dict[query.id()],
                        all_hard_negative_idxs=all_hard_negative_idxs,
                        similarity_scores_1d=similarity_scores[q_idx],
                        top_k_idxs_1d=top_k_idxs[q_idx]
                    )

                loss = batch_total_loss / len(cur_queries)
                print(f"Loss: {loss}")
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        # take it out of training mode
        self.question_encoder.eval()

    def encode_queries(self,
                       queries: List[Question],
                       batch_size: int = 16,
                         **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        with torch.no_grad():
            tokenized_questions = self.question_tokenizer([query.text() for query in queries], padding=True, truncation=True, return_tensors='pt').to("cuda")

        token_emb =  self.question_encoder(**tokenized_questions)
        # print("token_emb",token_emb[0].shape)
        sentence_emb = self.mean_pooling(token_emb[0],tokenized_questions["attention_mask"])
        # print("sentence_emb",sentence_emb.shape)
        assert sentence_emb.shape[0] == len(queries)
        return sentence_emb

    def save_query_encoder(self, save_directory : str) -> None:
        """
        Save fine-tuned model
        Args:
            save_directory: The directory where we want to save it, it has to exist, otherwise exception

        Returns: None
        """
        self.question_encoder.eval()
        self.question_encoder.save_pretrained(save_directory)

    def load_query_encoder(self, load_directory: str) -> None:
        """
        Load fine-tuned model
        Args:
            load_directory: The directory where it is saved, it has to exist, otherwise exception

        Returns: None
        """
        self.question_encoder = AutoModel.from_pretrained(load_directory)