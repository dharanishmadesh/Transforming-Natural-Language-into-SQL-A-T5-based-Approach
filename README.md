

# SQL Query Generation using T5 Model

This project demonstrates how to train a T5-based model to generate SQL queries from natural language questions. The model is trained on a dataset of questions and corresponding SQL queries and is capable of converting questions about a user's profile into structured SQL queries that can be used to retrieve data from a database.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Training the Model](#training-the-model)
5. [Evaluating the Model](#evaluating-the-model)
6. [Usage](#usage)
7. [Results](#results)
8. [License](#license)

## Overview

The project uses a **T5 model** (a transformer-based architecture) to convert natural language questions into SQL queries. The T5 model is fine-tuned on a dataset consisting of questions and corresponding SQL queries about a user's profile. This allows the model to generate SQL queries for different types of questions, such as retrieving a user's full name, date of birth, skills, achievements, and more.

The workflow involves:

1. Preparing the dataset (questions and SQL queries).
2. Fine-tuning the T5 model on the dataset.
3. Evaluating the model's performance using BLEU score.
4. Providing a function to respond to queries based on a user profile.

## Installation

To use this project, you need to install the following dependencies:

```bash
pip install pandas torch transformers nltk scikit-learn
```

## Dataset

The dataset contains pairs of **questions** and **SQL queries**. Each question represents a request about a user's profile, and the corresponding SQL query retrieves the relevant information from a database. 

Example questions:

- "What is the user's full name?"
- "What programming languages does the user know?"
- "Which IoT projects has the user completed?"

The SQL queries associated with the questions are designed to retrieve the appropriate information from a structured database.

## Training the Model

The model is based on the `t5-small` variant of the T5 architecture. The model is trained for 3 epochs with a batch size of 4. The training process involves:

1. Tokenizing the input questions and the corresponding SQL queries.
2. Feeding these tokens into the T5 model.
3. Backpropagating the loss and updating the model's weights.

Training is done using the **AdamW optimizer** with a learning rate of `5e-5`.

### Code to train the model:

```python
def train_model():
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 3

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
```

## Evaluating the Model

The model's performance is evaluated using the **BLEU score**, which measures the similarity between the generated SQL queries and the actual SQL queries in the test set.

### Code to evaluate the model:

```python
def evaluate_model():
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(len(X_test)):
            question = X_test.iloc[i]
            input_tokens = tokenizer(question, return_tensors='pt', padding=True, truncation=True)

            # Generate SQL query
            output = model.generate(**input_tokens)
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions.append(decoded_output)

    bleu_scores = []
    for pred, true in zip(predictions, y_test):
        bleu_scores.append(sentence_bleu([true.split()], pred.split()))  # Split and compute BLEU score

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu_score:.2f}")
```

## Usage

Once the model is trained, it can be used to generate SQL queries for any natural language question related to the user profile. The model is saved after training and can be reloaded for inference.

Here’s an example of how to use the model:

```python
# Example usage to respond to a query
query = "full_name"
response = respond_to_query(query)
print(query, ":", response)

query = "dob"
response = respond_to_query(query)
print(query, ":", response)

query = "gender"
response = respond_to_query(query)
print(query, ":", response)

query = "degree"
response = respond_to_query(query)
print(query, ":", response)

# ... More queries
```

You can replace the query with any valid question from the dataset to get a response.

## Results

The model's performance is measured by the **BLEU score**. A higher BLEU score indicates that the model’s generated SQL queries are closer to the actual queries in the test set.

Example output:
```
Average BLEU score: 0.45
```

## License

This project is licensed under the MIT License 
