from dexter.config.constants import Split
from dexter.data.datastructures.answer import AmbigNQAnswer, Answer
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.sample import AmbigNQSample
from dexter.data.loaders.BaseDataLoader import GenericDataLoader
import tqdm


class MyDataLoader(GenericDataLoader):
    
    def __init__(
        self,
        dataset: str,
        tokenizer="bert-base-uncased",
        config_path=None,
        split=Split.TRAIN,
        batch_size=None,
        corpus=None
    ):
        self.corpus = corpus
        self.config_path = config_path
        super().__init__(dataset, split)
        
    def load_raw_dataset(self, split):
        dataset = self.load_json(split)
        for i in tqdm.tqdm(range(len(dataset))):
            sample = dataset[i]
            question = Question(sample['question'])
            evidences = []
            for evidence in sample['evidences']:
                evidences.append(Evidence(evidence[0], evidence))
            answers = Answer(sample['answer'], None)
            self.raw_data.append(AmbigNQSample(question, evidences, answers))



