import io
import tensorflow as tf

lines = io.open("data\SMSSpamCollection",encoding='utf-8').read().strip().split("\n")
# ham / spam
# print(lines[0])

# Set up labels 0, 1
spam_dataset = []
for line in lines:
    label, text = line.split("\t")
    if label.strip() == "spam":
        spam_dataset.append((1, text.strip()))
    else:
        spam_dataset.append((0, text.strip()))
# print(spam_dataset[0])


# Text Normalization
import pandas as pd
df = pd.DataFrame(spam_dataset, columns=["Spam", "Message"])

# regular expression
import re
def message_length(x):
    return len(x)

def num_capitals(x):
    _, count = re.subn(r"[A-Z]", "", x)
    return count

def num_punctuation(x):
    _, count = re.subn(r"\W", "", x)
    return count

df["Capitals"] = df["Message"].apply(num_capitals)
df["Punctuation"] = df["Message"].apply(num_punctuation)
df["Length"] = df["Message"].apply(message_length)
# print(df.describe())

# split training and testing sets
train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

x_train = train[["Length", "Capitals", "Punctuation"]]
y_train = train["Spam"]
x_test = test[["Length", "Capitals", "Punctuation"]]
y_test = test["Spam"]

# 1-layer neural network model for evaluation
def make_model(input_dim=3, num_units=12):
    model = tf.keras.Sequential()
    # Adds densely-connected layer
    model.add(tf.keras.layers.Dense(num_units, input_dim=input_dim, activation="relu"))
    # Add sigmoid layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

model = make_model()
model.fit(x_train, y_train, epochs=10, batch_size=10)

# prediction
y_train_pred = model.predict_classes(x_train)
# confusion matrix
print(tf.math.confusion_matrix(tf.constant(y_train), y_train_pred))


res = model.evaluate(x_test, y_test)
print("Acc: ", res[1])