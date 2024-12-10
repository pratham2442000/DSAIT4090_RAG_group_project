# DSAIT4090_RAG_group_project

For data 

- Go to [link](https://drive.google.com/drive/folders/1aQAfNLq6HB0w4_fVnKMBvKA6cXJGRTpH)
- Download all the files, (i.e all the .json files) individually and put them in the data/ folder
- Your data folder should now look like

```bash
└── data
    ├── dev.json
    ├── test.json
    ├── train.json
    └── wiki_musique_corpus.json
```

For setup follow the instructions in README_dexter.md, specifically:
- Create a conda environment conda create -n bcqa
- pip install -e .

Note: use python version 3.10

Our dataset follows the format of data.
Each sample has the following keys:
- ```_id```: a unique id for each sample
- ```question```: a string
- ```answer```: an answer to the question. The test data does not have this information.
- ```supporting_facts```: a list, each element is a list that contains: ```[title, sent_id]```, ```title``` is the title of the paragraph, ```sent_id``` is the sentence index (start from 0) of the sentence that the model uses. The test data does not have this information.
- ```context```: a list, each element is a list that contains ```[title, setences]```, ```sentences``` is a list of sentences.
- ```evidences```: a list, each element is a triple that contains ```[subject entity, relation, object entity]```. The test data does not have this information.
- ```type```: a string, there are four types of questions in our dataset: comparison, inference, compositional, and bridge-comparison.
- ```entity_ids```: a string that contains the two Wikidata ids (four for bridge_comparison question) of the gold paragraphs, e.g., 'Q7320430_Q51759'.
