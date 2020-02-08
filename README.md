This is an abstractive summarization system based on the [Pytorch](https://github.com/pytorch/pytorch) port of [OpenNMT](https://github.com/OpenNMT/OpenNMT), an open-source (MIT) neural machine translation system. 

The system is designed to explore pre-training and locality modeling applied to Transformer and consists of two new models:
* BERT-transformer - the model using pre-trained [BERT](https://github.com/huggingface/transformers) to condition encoder and decoder of vanilia Transformer.
* Convolutional Transformer - the model replacing self-attention with convolutional self-attention.

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```
## Usage
-bert_multilingual
-bert,-bert-transformer,-conv-transformer
-bert



