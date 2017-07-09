# Generative NLP with Reinforcement learning
Code for reproducing main results in the paper [A Neural Conversational Model](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) to build a conversational chat bot using seq2seq model.

This model extends seq2seq with these features:
- Reinforcment learning for more coherent dialouge instead of 1-to-1 relpy.
- Buckets while training to limit memory consumption
- Beam search to avoid local minima resopnses (needs to be optimized)
- Anti-langauge moddel to supress global solutions like "Thanks, ok" when uncertain.

<img src="Images/seq2seq.png" width="800px"/>


### Dependencies
- python 3.5
- [TensorFlow 1.0 version](https://www.tensorflow.org/get_started/os_setup)
- asyncio
- websockets

`
pip install -r requirements.txt
`


### Data
1. The format should be one txt file with input on a line followed by the response on the next line.

    ```
    Datasets/
    |
    |-Your-dataset/
      |
      |-data/raw/
        |- chat.txt

    twitter             Twitter chat by [Marsan](https://github.com/Marsan-Ma/chat_corpus)
    ```

2. Optional: to get trained model, download and extract:
    - [twitter checkpoint](https://github.com/Roboy/roboy_generative_nlp/tree/datasets)


### Training
- To train Seq2Seq run `seq2seq_train` passing model name
   
    ```
    python seq2seq_train.py --model_name twitter and pass in params
    ```


### Evaluation
- To evaluate Seq2Seq run `seq2seq_eval` passing model name

    ```
    python seq2seq_eval.py --model_name twitter and pass in params
    ```

- Or simply run ./twitter_chat which will download the trained model and start the chatting interface. This model wasn't trained with reinforcment learning as the chat dataset is not dialouge.

    ```
    ./twitter_chat
    ```

### Papers
[\[1\] Sequence to Sequence Learning with Neural Networks][1]
[\[2\] A Neural Conversational Model][2]

[1]: http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

[2]: http://arxiv.org/pdf/1506.05869v1.pdf
