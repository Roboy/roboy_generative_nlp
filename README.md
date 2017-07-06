# Trained models with Seq2Seq NLP

### Twitter chat
This datasets wasn't trained with reinforcment learning as the chat data is not continous (dialouge).

- vocab_size 100000
- layer_size 128
- layers_num 4

### Extracting
- Download compressed files then extract them into Datasets folder
   
    ```
    cat twitter.tar.gz.part-* > twitter.tar.gz
    tar -zxvf twitter.tar.gz
    rm twitter.tar.gz*
    ```