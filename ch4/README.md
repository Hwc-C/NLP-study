# Chapter 4. Transfer Learning with BERT

GloVe pre-trained embeddings + BERT

## Transfer learning overview

- Knowledge distillation, pre-training
- Adaptation, fine-tuning

Three main types of transfer learning, 
- domain adaptation
- multi-task learning
- sequential learning

## Domain Adaptation
Dealing with similar situations. 

Assume training and testing data are *i.i.d*, but it is frequently violated. Some techniques could mitigate the gap between training and potential testing data. 

## Multi-task learning
Increases the amount of data available for training by pooling data from different tasks together. 

## Sequential learning
- The first step, pre-train model
- The second step, load pre-trained model (frozen/update/fine-tuned). 

## GloVe embeddings
Takes the frequencies into account and posits that the co-occurrences provide vital info. 
GloVe focuses on the ratios of co-occurrence considering probe words. 

*Example*:
$$\frac{P_{solid|ice}}{P_{solid|steam}}$$


## BERT-based transfer learning
Two foundational advancements enabled BERT, 
- encoder-decoder network
- Attention mechanism

## Encoder-decoder networks
Input tokens -> LSTM -> TimeDistributed() -> Output Dense()

Translation-type tasks, input and output may have different length. The solution, refer to as the seq2seq. 

## Attention model
The Attention mechanism allows te decoder part of the network to see the encoder hidden states. **General Attention**, decoder operates on a sequence of vectors generating by encoding the input rather than one fixed vector generated at the end of the input. **Self-attention** enables connections between different encodings of input tokens in different positions. **Bahdanau Attention** enables each output state to look at the encoded inputs and learn some weights for each of these inputs. 

## Transformer model
**Language Model** task is traditionally defined as predicting the next word in a sequence of words. LMs are particularly useful for text generation, but less for classification. 

**Attention layer**, value vectors of tokens with higher softmax scores will have higher contribution to the output value vector of the input token in question. 

**Multi-head self-attention**, creates multiple copies of the query, key and value vectors along with the weights matrix used to compute the query from the embedding of the input token. An additional weight matrix is used to combine the multiple outputs of each of the heads and concatenate them together into one output value vector. 

**Output value vector**, is fed to the feed-forward layer, and the output of the feed-forward layer goes to the next encoder block or becomes the output of the model at the final encoder block. 

## The bidirectional encoder representations from transformers (BERT) model

*BERT base*, 12 encoder blocks + 12 attention heads + 768 hidden layers.

**Masked Language Model** (MLM), some of the input tokens are masked randomly, model has to predict the right token given the tokens on both sides of the masked token. 

**Next Sentence Prediction** (NSP), deal with pairs of sentences (question-answering problem). 

BERT addresses a problem out-of-vocabulary tokens. BERT uses the WordPiece tokenization scheme with a vocabulary size of 30,000 tokens, smaller dictionary. 
- subword
- Byte Pair Encoding (BPE)
- SentencePiece
- the Unigram

A minor adjustment to the model RoBERT. 

What happen in BERT:

> **Sentence**::This was an absolutely terrible movie. Don't be **lured** in by Christopher Walken or Michael Ironside.
> 
> **After tokenization**::[CLS] This was an absolutely terrible movie . Don' t be **lure ##d** in by Christopher Walk ##en or Michael Iron ##side . [SEP]

- [CLS], the start of the inputs
- [SEP], be added in between the sequences

```python
!pip install transformers

from transformers import BertTokenizer
bert_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(
    bert_name,
    add_special_tokens=True,
    do_lower_case=False, 
    max_length=150,
    pad_to_max_length=True
    )
```

Three sequences: 
- input_ids, tokens converted into IDs
- token_type_ids, about the segment identifiers
- attention_mask, tell the model where the actual tokens end

Merge into dict(), then train. 
```python
def example_to_features(input_ids,attention_masks,token_type_ids,y):
    return {"input_ids": input_ids,
        "attention_mask": attention_masks,
        "token_type_ids": token_type_ids},y

train_ds = tf.data.Dataset.from_tensor_slices((tr_reviews,
    tr_masks, tr_segments, y_train)).\
    map(example_to_features).shuffle(100).batch(16)
```

## Custom model with BERT

```python
from transformers import TFBertModel
bert_name = 'bert-base-cased'
bert = TFBertModel.from_pretrained(bert_name)
bert.summary()
```

```python
inp_dict = {"input_ids": inp_ids,
    "attention_mask": att_mask,
    "token_type_ids": seg_ids}
outputs = bert(inp_dict)
# let's see the output structure
outputs
```
The first output has embeddings for each of the input tokens includin the special tokens. The second output corresponds to the output of the [CLS] token. The output[1] will be used further in the model. 




