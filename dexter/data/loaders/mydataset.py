from dexter.config.constants import Split
from dexter.data.datastructures.answer import Answer
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.evidence import Evidence
from dexter.data.datastructures.sample import Sample
from dexter.data.loaders.BaseDataLoader import GenericDataLoader
import tqdm


class MyDataLoader(GenericDataLoader):
    
    def __init__(
        self,
        dataset: str,
        tokenizer=None,
        config_path=None,
        split=Split.TRAIN,
        batch_size=None,
        corpus=None
    ):
        self.corpus = corpus
        self.config_path = config_path
        super().__init__(dataset, tokenizer, config_path, split, batch_size)
        
    def load_raw_dataset(self, split):
        dataset = self.load_json(split)
        print("loading mydataset")
        for index,i in enumerate(tqdm.tqdm(range(len(dataset)))):
            sample = dataset[i]
            question = Question(sample['question'],index)
            evidences = []
            for evidence in sample['evidences']:
                evidences.append(Evidence(title=evidence[0], text=evidence, idx=index))
            answers = Answer(sample['answer'], index)
            self.raw_data.append(Sample(idx=index,question=question, evidences=evidences, answer=answers))
        print("dataset loaded, moving on to other init")


