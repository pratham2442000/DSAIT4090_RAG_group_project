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




if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="facebook/contriever",
                                     document_encoder_path="facebook/contriever"
                                     ,batch_size=32,show_progress_bar=True)

    loader = RetrieverDataset("data","wiki_musique_corpus","config.ini",Split.DEV,tokenizer=None)
    queries, qrels, corpus = loader.qrels()
    
    
    con = Contriever(config_instance)


    similarity_measure = CosScore()
    response = con.retrieve(corpus,queries,100,similarity_measure)
    print("indices",len(response))
    metrics = RetrievalMetrics(k_values=[1,3,5])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))

# %%



