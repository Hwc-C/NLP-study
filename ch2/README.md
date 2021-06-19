# Chapter 2. Understanding Sentiment in Natural Language with BiLSTMs

## **Natural language understanding**

NLU enables the processing of unstructureed text and extracts meaning and critical pieces of information that are actionable. 

## **Bi-directional LSTMs - BiLSTMs**
Using the output generated after processing the prevous item in the sequence along with the current item to generate the next output. 

## RNN building blocks

$$a_t=Ux_t+Va_{t-1}$$

$$o_t=Wa_t$$

## Long short-term memory networks

Four main parts:
- Cell value
- Input gate
- Output gate
- Forget gate

## Gated recurrent units

Input and forget gates are combined into a single update gate. 

## **Sentiment classification with LSTMs**

Using *tensorflow_datasets* to get dataset. 

```
!pip install tensorflow_datasets

import tensorflow_datasets as tfds

# Example
imdb_train, ds_info = tfds.load(name="imdb_reviews", split="train", with_info=True, as_supervised=True)

# Check the first sample
for example, label in imdb_train.take(1):
	print(example, "\n", label)
```

```
# Encode the words using the vocabulary
tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
MAX_TOKENS = 0

for example, label in imdb_train:
	some_tokens = tokenizer.tokenize(example.numpy())
	if MAX_TOKENS < len(some_tokens):
		MAX_TOKENS = len(some_tokens)
	vocabulary_set.update(some_tokens)
```

What happen: 

Get a set of data with index, based on tokenizer function. 

```
# save
imdb_encoder.save_to_file("reviews_vocab")

# load
enc = tfds.features.text.TokenTextEncoder.load_from_file("reviews_vocab")
enc.decode(enc.encode("Good case. Excellent value."))
```

- Tokenization, to be tokenized into words
- Encoding, map to integers using the vocabulary
- Padding, variable lengths

```
from tensorflow.keras.preprocessing import sequence

def encode_pad_transform(sample):
	encoded = imdb_encoder.encode(sample.numpy())
	pad = sequence.pad_sequences([encoded], padding="post, maxlen=150)
	return np.array(pad[0], dtype=np.int64)

def encode_tf_fn(sample, label):
	encoded = tf.py_function(encode_pad_transform, inp=[sample], Tout=(tf.int64))
	encoded.set_shape([None])
	label.set_shape([])
	return encoded, label

encoded_train = imdb_train.map(encode_tf_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
encoded_test = imdb_test.map(encode_tf_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
```

## LSTM and BiLSTM model with embeddings

```
tf.keras.layers.LSTM(rnn_units)

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units))
```


Problem: Overfitting

with some dropout for regularization

using the unsupervised split of the dataset

adding more features such as word shapes, POS tags
