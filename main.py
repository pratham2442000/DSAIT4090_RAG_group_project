# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: dsait4090
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Contriever Inference

# %%
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.retriever.dense.Contriever import Contriever
from dexter.config.constants import Split
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity as CosScore
from dexter.utils.metrics.CoverExactMatch import CoverExactMatch
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams

# from importlib import reload
# reload(RetrieverDataset)


# %%

config_instance = DenseHyperParams(query_encoder_path="facebook/contriever",
                                    document_encoder_path="facebook/contriever"
                                    ,batch_size=32,show_progress_bar=True)

loader = RetrieverDataset("wikimultihopqa","wiki_musique_corpus","config.ini",Split.DEV,tokenizer=None)
queries, qrels, corpus = loader.qrels()


con = Contriever(config_instance)


similarity_measure = CosScore()
response = con.retrieve(corpus,queries,100,similarity_measure)
print("indices",len(response))
metrics = RetrievalMetrics(k_values=[1,3,5])
print(metrics.evaluate_retrieval(qrels=qrels,results=response))
