## Neural-based Abstractive Text Summarization

This system implements the abstractive text summarization models from the paper [Abstractive Text Summarization based on Language Model Conditioning and Locality Modeling
](https://arxiv.org/abs/2003.13027). 

The system supports two neural models:
* BERT-Transformer (bert) - the model using pre-trained [BERT](https://github.com/huggingface/transformers) to condition the encoder and decoder of Transformer.
* Transformer with Convolutional Self-Attention (conv) - the model replacing self-attention with convolutional self-attention to better model local dependencies.

For summarization of long texts, the TF-IDF extractive summarizer can be used before the abstractive models.

## Usage
First, download the models from [here](https://drive.google.com/file/d/1dDhfbRneUUNfVEMWB-fEpSqZz8VwsjID/view?usp=sharing) and extract them in the *models/* folder.

Then, run the system specifyning the language of the text (English and German), the method of summarization and if the extractive summarizer must be used before the abstractive one.

The example of usage:
```python
from summarizer import AbstractiveSummarizer
texts = []
with open("data/sample_en.txt") as f:
	texts = [text for text in f]

model = AbstractiveSummarizer(language = 'en', method = 'conv', extract = True)
for summ in model.summarize(texts):
	print(summ)
```
