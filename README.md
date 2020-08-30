This is the implimentation of the paper [Abstractive Text Summarization based on Language Model Conditioning and Locality Modeling
](https://arxiv.org/abs/2003.13027). It is developed as a fork of [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) with two new models:

* BERT-Transformer - the model using pre-trained [BERT](https://github.com/huggingface/transformers) to condition encoder and decoder of Transformer.
* Convolutional Transformer - the model replacing self-attention with convolutional self-attention to better model local dependencies.


## Usage
To use the system the data must be tokenized by [BERT tokenizer](https://github.com/huggingface/transformers) and saved in the text tile, with tockens separated by spaces and and one line per text.

For general documentation check original [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) repository.

To train and use the system model follow the [documentation](https://opennmt.net/OpenNMT-py/Summarization.html) of OpenNMT for summarizaiton.
* To use BERT-based model add *-encoder_type bert-transformer* and *-decoder_type bert* to the train and test scripts parameters
* To use the Convolutional Transformer model add *-encoder_type conv-transformer*
* For non English data add *-bert_multilingual*
