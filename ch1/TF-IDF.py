import io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


lines = io.open("data\SMSSpamCollection",encoding='utf-8').read().strip().split("\n")

spam_dataset = []
for line in lines:
    label, text = line.split("\t")
    if label.strip() == "spam":
        spam_dataset.append((1, text.strip()))
    else:
        spam_dataset.append((0, text.strip()))

df = pd.DataFrame(spam_dataset, columns=["Spam", "Message"])

train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

transformer = TfidfVectorizer(binary=True)
X = transformer.fit_transform(train["Message"]).astype("float32")
X_test = transformer.transform(test["Message"]).astype("float32")

# print(X.shape)
# print(X.toarray())

# train model
import tensorflow as tf
def make_model(input_dim=3, num_units=12):
    model = tf.keras.Sequential()
    # Adds densely-connected layer
    model.add(tf.keras.layers.Dense(num_units, input_dim=input_dim, activation="relu"))
    # Add sigmoid layer
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

# input_dim
model = make_model(X.shape[1])

y_train = train[["Spam"]]
y_test = test[["Spam"]]

model.fit(X.toarray(), y_train, epochs=10, batch_size=10)
model.evaluate(X_test.toarray(), y_test)

# confusion matrix
y_test_pred = model.predict_classes(X_test.toarray())

print("Confusion matrix:\n", tf.math.confusion_matrix(tf.constant(y_test.Spam),y_test_pred))
