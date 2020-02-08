This is an abstractive summarization system based on the [Pytorch](https://github.com/pytorch/pytorch) port of [OpenNMT](https://github.com/OpenNMT/OpenNMT), an open-source (MIT) neural machine translation system. 

The system is designed to explore pre-training and locality modeling applied to Transformer and consists of two new models:
* BERT-Transformer - the model using pre-trained [BERT](https://github.com/huggingface/transformers) to condition encoder and decoder of vanilia Transformer.
* Convolutional Transformer - the model replacing self-attention with convolutional self-attention.

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```
## Usage
To use the system the data must be tokenized by [WordPiece tokenizer](https://github.com/huggingface/transformers) and saved in the OpenNMT input format.

For general documentation check original [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) repository.

To train the BERT-Transformer model set -encoder_type to bert-transformer and -decoder_type to bert. For non English data add -bert_multilingual property.

To train the Convolutional Transformer model set -encoder_type to conv-transformer.
