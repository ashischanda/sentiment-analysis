# Sentiment analysis of Twitter data to predict disaster
We conducted a research work to analysis sentiment of Twitter data based on BOW, contextual and context-free embeddings to predict disaster. We used three traditional machine learing models (decision tree, random forest, and logistic regression) and three popular pre-trained contextual embedings (Skip-gram, FastText, and GloVe). For context-free embeddings, we used pre-trained BERT (Bert-base-uncased) model.

## Download data
We used data from a Kaggle competition. To download the data, please visit the following link:
https://www.kaggle.com/c/nlp-getting-started


## Download pre-trained embeddings
To download pre-trained embeddings, please visit the following link:
1) https://nlp.stanford.edu/projects/glove/
2) https://github.com/google-research/bert
3) "Advances in Pre-Training Distributed Word Representations", Mikolov T. G. and et al. Proceedings of the International Conference on Language Resources and Evaluation, LREC 2018

## Reproduce our result
To reproduce the experimental results, follow the following steps:
1) Download data from the Kaggle competition and keep in "data" folder.
2) Download the pre-trained embeddings from the above folder and keep in the same source folder. Write the file name in our python code to load the embeddings. For example, you need to update the GLOVE_EMB variable in GloVe_softmax.py file with your own GloVe embedding file name.

## Citation
Please acknowledge the following work in papers or derivative software:

@inproceedings{akc,
    title = "Sentiment analysis of Twitter data to predict disaster",
    author = "AKC",
    year = "2021",
    publisher = "",
    url = "",
    doi = "",
    pages = "1-8"
}
