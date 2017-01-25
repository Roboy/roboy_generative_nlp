# Generative NLP
Code for reproducing main results in the paper [A Neural Conversational Model](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) to build a conversational chat bot using seq2seq model.

<img src="Images/seq2seq.png" width="800px"/>


### Dependencies
- python 2.7
- [TensorFlow 0.12 version](https://www.tensorflow.org/get_started/os_setup)
- In addition, please `pip install -r requirements.txt` to install the following packages:
    - `nltk`
    - `pickle`


### Data
1. Datasets are available in datasets folder. The format should be one file with input on a line followed by the response on the next line.
    ```
    Squad               Stanford Q&A dataset (100,000+)
    wiki_qa             Short wikipedia Q&A (3,400+)
    twitter             Twitter chat by [Marsan](https://github.com/Marsan-Ma/chat_corpus)
    ```

2. Preprocess and filter raw dataset using

    ```
    python data/data_preprocess.py --file_path Datasets/squad/squad.txt --output_dir Datasets/squad/processed and pass in params
    ```
    The data is filtered by removing too long or too short sequences, only keeping the most frequent vocab size(8000) words from the data. the data is saved as q ids & a ids & meta data holding words.

3. Optional: to get trained model, download and extract:
    - [twitter checkpoint](https://transfer.sh/12XXye/twitter-checkpoint.tar.gz)


### Training
- To train Seq2Seq run `seq2seq_train` passing processed dataset dir and checkpoint dir (to save model)
   
    ```
    python seq2seq_train.py --dataset_dir Datasets/squad --ckpt_dir Datasets/squad/checkpoints and pass in params
    ```


### Evaluation
- To evaluate Seq2Seq run `seq2seq_eval` passing processed dataset dir and checkpoint dir (to load model)

    ```
    python seq2seq_eval.py --dataset_dir Datasets/squad --ckpt_dir Datasets/squad/checkpoints and pass in params
    ```


### Predict
- To test Seq2Seq with cmd run `seq2seq_predict` passing processed dataset dir and checkpoint dir (to load model)

    ```
    python seq2seq_predict.py --dataset_dir Datasets/squad --ckpt_dir Datasets/squad/checkpoints and pass in params
    ```


### Papers
[\[1\] Sequence to Sequence Learning with Neural Networks][1]
[\[2\] A Neural Conversational Model][2]

[1]: http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

[2]: http://arxiv.org/pdf/1506.05869v1.pdf
