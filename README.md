
# Chatbot Using LSTM

This project implements a simple chatbot using Long Short-Term Memory (LSTM) networks. It is built with TensorFlow and Keras to process a dataset of questions and answers. The chatbot is trained to generate responses based on input sequences.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- TensorFlow
- Keras
- Numpy
- Pandas
- Scikit-learn

You can install the required packages using the following command:

```bash
pip install tensorflow keras numpy pandas scikit-learn
```

## Files Included

- **extended_chat_data.csv**: The CSV file containing questions and answers used for training.
- **chatBot.h5**: The trained chatbot model saved in HDF5 format after training.

## Steps to Run

### 1. Data Preparation

The `extended_chat_data.csv` file is read, and the questions and answers are tokenized. The sequences are padded to ensure uniform length.

```python
data = pd.read_csv('extended_chat_data.csv', on_bad_lines='skip')
questions = data['Question'].values
answers = data['Answer'].values
```

### 2. Tokenization

We use the Keras `Tokenizer` to convert questions and answers into sequences of integers. These sequences are padded to have the same length.

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
questions_seq = tokenizer.texts_to_sequences(questions)
answers_seq = tokenizer.texts_to_sequences(answers)
```

### 3. Train-Test Split

The data is split into training and testing sets using an 80-20 split.

```python
x_train, x_test, y_train, y_test = train_test_split(questions_seq, answers_seq, test_size=0.2)
```

### 4. Model Architecture

The chatbot uses a Sequential model with the following layers:

- **Embedding Layer**: Converts input tokens into dense vectors of fixed size.
- **LSTM Layer**: Processes the input sequences.
- **Dense Layer**: Outputs the probability distribution of words to be generated.

```python
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_seq_len))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(vocab_size, activation='softmax'))
```

The model is compiled using `sparse_categorical_crossentropy` as the loss function and trained with the Adam optimizer.

### 5. Training

If the model does not already exist as `chatBot.h5`, it is trained on the training data for 200 epochs with a batch size of 32.

```python
model.fit(x_train, y_train_int, epochs=200, batch_size=32, validation_data=(x_test, y_test_int))
```

### 6. Response Generation

The chatbot generates a response by predicting the next word based on the input sequence.

```python
def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_seq_len)
    prediction = model.predict(input_seq)
    response_seq = np.argmax(prediction, axis=-1)
    response = tokenizer.sequences_to_texts([response_seq])
    return response[0]
```

### 7. Example Conversation

```bash
You: Hello
Bot: Hi, how can I help you?

You: What is your name?
Bot: I am a chatbot.

You: quit
```

## Usage

1. Clone or download this repository.
2. Place your `extended_chat_data.csv` in the project folder.
3. Run the script to train the chatbot:

```bash
python chatbot_script.py
```

4. You can then interact with the chatbot in the console. Type "quit" to exit.

## Model Saving and Loading

The trained model is saved as `chatBot.h5` and can be loaded for future use.

```python
model = load_model("./chatBot.h5")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
