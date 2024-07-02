import os
import re
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Clear cached memory
torch.cuda.empty_cache()
import sklearn_crfsuite

def tokenize(sentence):
    # Simple tokenizer that can handle basic punctuation
    return re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def train_crf(X_train, y_train):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    return crf

def convert_to_bilou(context, labels):
    words = tokenize(context)
    bilou_labels = ["O"] * len(words)
    print(f"Context Words: {words}")

    for label in labels:
        label_parts = label.split(maxsplit=1)
        if len(label_parts) < 2:
            print(f"Skipping invalid label: {label}")
            continue

        label_type, label_text = label_parts
        label_words = tokenize(label_text)
        print(f"Label Type: {label_type}, Label Words: {label_words}")

        match_found = False
        for i in range(len(words) - len(label_words) + 1):
            if words[i:i + len(label_words)] == label_words:
                match_found = True
                if len(label_words) == 1:
                    bilou_labels[i] = f"U-{label_type}"
                else:
                    bilou_labels[i] = f"B-{label_type}"
                    for j in range(1, len(label_words) - 1):
                        bilou_labels[i + j] = f"I-{label_type}"
                    bilou_labels[i + len(label_words) - 1] = f"L-{label_type}"
                break
        if not match_found:
            print(f"No match found for '{label_text}' in {words}")

    print(f"BILOU Labels: {bilou_labels}")
    return bilou_labels

# Load your datasets efficiently
dataset = load_dataset('csv', data_files={
    'train': '/teamspace/studios/this_studio/train_subtask2.csv',
    'validation': '/teamspace/studios/this_studio/dev_subtask2.csv'
}, cache_dir='./cache')
test_dataset = load_dataset('csv', data_files={
    'test': '/teamspace/studios/this_studio/test_subtask2_text.csv'
}, cache_dir='./cache')
def prepare_data_for_crf(dataset):
    X_data = []
    y_data = []
    for example in dataset:
        tokens = tokenize(example['text'])
        # Ensure 'labels' exist and are provided correctly, fallback to an empty list if not present
        bilou_labels = convert_to_bilou(example['text'], example.get('labels', []))

        # Generate features for each token in the document
        X_features = sent2features(tokens)
        # Append features and labels for this example to the dataset lists
        X_data.append(X_features)
        y_data.append(bilou_labels)

    return X_data, y_data

# Prepare and train the CRF
X_train, y_train = prepare_data_for_crf(dataset['train'])
crf_model = train_crf(X_train, y_train)

# Function to print GPU memory usage
def print_gpu_utilization():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3} GB")
# Example usage in the training loop


# Step 4: Causal Component Identification using T5 with Gradient Checkpointing
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_model.gradient_checkpointing_enable()

# Ensure use_cache=False when using gradient checkpointing
t5_model.config.use_cache = False
# Specify the path to save the model
save_directory = "./saved_model"

# Save the model and tokenizer
t5_model.save_pretrained(save_directory)
t5_tokenizer.save_pretrained(save_directory)

def preprocess_t5_function(examples):
    inputs = ["extract causality: " + text for text in examples['text']]
    model_inputs = t5_tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    contexts = examples['text']
    raw_labels = examples['text_w_pairs'] if 'text_w_pairs' in examples else ["" for _ in contexts]
    labels = [convert_to_bilou(context, [raw_label]) if raw_label else ["O"] * len(context.split()) for context, raw_label in zip(contexts, raw_labels)]
    labels = [" ".join(label) for label in labels if label]
    
    labels = t5_tokenizer(labels, max_length=128, truncation=True, padding="max_length").input_ids
    
    labels = [[label if label != t5_tokenizer.pad_token_id else -100 for label in labels_instance] for labels_instance in labels]
    
    model_inputs["labels"] = labels
    return model_inputs


# Prepare data for T5
def test_preprocess_t5_function(examples):
    inputs = ["extract causality: " + text for text in examples['text']]
    model_inputs = t5_tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    # Ensure labels are strings
    labels = [str(label) for label in examples['causal_text_w_pairs']]
    labels = t5_tokenizer(labels, max_length=128, truncation=True, padding="max_length").input_ids
    
    # Replace padding tokens with -100 for labels
    labels = [[label if label != t5_tokenizer.pad_token_id else -100 for label in labels_instance] for labels_instance in labels]
    
    model_inputs["labels"] = labels
    return model_inputs

# Preprocess the datasets
encoded_train_dataset = dataset['train'].map(preprocess_t5_function, batched=True)
encoded_val_dataset = dataset['validation'].map(preprocess_t5_function, batched=True)
encoded_test_dataset = test_dataset['test'].map(test_preprocess_t5_function, batched=True)

# Define training arguments with early stopping
training_args = TrainingArguments(
    output_dir='./t5_results',
    evaluation_strategy='steps',
    eval_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,  # Increase epochs but use early stopping
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
    no_cuda=False,
    dataloader_num_workers=0,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

def compute_t5_metrics(p):
    predictions = p.predictions
    labels = p.label_ids
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    if len(predictions.shape) > 2:
        predictions = np.argmax(predictions, axis=-1)
    
    decoded_preds = [t5_tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_labels = []
    for label in labels:
        filtered_label = [token for token in label if token != -100]
        decoded_labels.append(t5_tokenizer.decode(filtered_label, skip_special_tokens=True))

    # Debug: Check for None or empty predictions and labels
    if any(pd is None or pd == '' for pd in decoded_preds) or any(dl is None or dl == '' for dl in decoded_labels):
        print("Warning: Found empty or None predictions/labels")
    
    # Calculate exact match accuracy
    exact_matches = [pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]
    accuracy = sum(exact_matches) / len(exact_matches) if len(exact_matches) > 0 else 0.0
    
    # Flatten predictions and labels
    flattened_preds = [token for sublist in decoded_preds for token in sublist.split() if token]
    flattened_labels = [token for sublist in decoded_labels for token in sublist.split() if token]

    # Debug: Check lengths
    if len(flattened_preds) != len(flattened_labels):
        print(f"Warning: Length mismatch between flattened_preds and flattened_labels - {len(flattened_preds)} vs {len(flattened_labels)}")

    min_length = min(len(flattened_preds), len(flattened_labels))
    flattened_preds = flattened_preds[:min_length]
    flattened_labels = flattened_labels[:min_length]

    # Calculate precision, recall, and F1 score
    precision = precision_score(flattened_labels, flattened_preds, average='weighted', zero_division=0)
    recall = recall_score(flattened_labels, flattened_preds, average='weighted', zero_division=0)
    f1 = f1_score(flattened_labels, flattened_preds, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Initialize the Trainer with early stopping callback
trainer = Trainer(
    model=t5_model,
    args=training_args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_val_dataset,
    tokenizer=t5_tokenizer,
    compute_metrics=compute_t5_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
trainer.train()
eval_results = trainer.evaluate()
print("Validation Evaluation Results:", eval_results)
# Move model to GPU
t5_model.to("cuda")

# Define function for generating predictions
def identify_causal_components(text):
    inputs = t5_tokenizer("extract causality: " + text, return_tensors="pt").to("cuda")
    outputs = t5_model.generate(**inputs, num_beams=4, early_stopping=True)
    decoded_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output if decoded_output else ""

# Generate predictions on test dataset
t5_predictions = []
contexts = [example['text'] for example in test_dataset['test']]
def predict_with_crf(crf_model, text):
    tokens = tokenize(text)
    features = sent2features(tokens)
    tags = crf_model.predict_single(features)
    return tags
for context in contexts:
    prediction = identify_causal_components(context)
    t5_predictions.append(prediction)
    bilou_tags = predict_with_crf(crf_model, context)
    #formatted_text = insert_spans(context, bilou_tags)

# Save the predictions to a CSV file
output_df = pd.DataFrame({
    'text': contexts,
    'causal_text_w_pairs': t5_predictions
})

output_df.to_csv('/teamspace/studios/this_studio/predicted_causal_text_w_pairs.csv', index=False)
print("Predictions saved to /teamspace/studios/this_studio/predicted_causal_text_w_pairs.csv")
def insert_spans(text, bilou_tags):
    words = tokenize(text)  # Make sure this tokenizer splits text the same way it's done in convert_to_bilou
    formatted_text = []
    open_tag = None

    # Define your tag mapping based on your specific needs
    tag_mapping = {
        'ARG0': 'ARG0',
        'ARG1': 'ARG1',
        'SIG': 'SIG0'  # Adjust this if your tagging scheme uses a different identifier for signals
    }

    for word, tag in zip(words, bilou_tags):
        if tag == 'O':
            if open_tag:
                formatted_text.append(f"</{open_tag}>")
                open_tag = None
            formatted_text.append(word)
        else:
            tag_type, label = tag.split('-') if '-' in tag else ('O', None)
            mapped_label = tag_mapping.get(label, label)  # Use the mapping to get the right tag

            if tag_type == 'B':
                if open_tag:
                    formatted_text.append(f"</{open_tag}>")
                open_tag = mapped_label
                formatted_text.append(f"<{mapped_label}>{word}")
            elif tag_type == 'I' and open_tag:
                formatted_text.append(f" {word}")
            elif tag_type == 'L' and open_tag:
                formatted_text.append(f" {word}</{mapped_label}>")
                open_tag = None
            elif tag_type == 'U':
                if open_tag:
                    formatted_text.append(f"</{open_tag}>")
                formatted_text.append(f"<{mapped_label}>{word}</{mapped_label}>")
                open_tag = None

    if open_tag:
        formatted_text.append(f"</{open_tag}>")

    return ' '.join(formatted_text)


# Process predictions to add spans
labeled_predictions = []
for context, prediction in zip(contexts, t5_predictions):
    bilou_tags = convert_to_bilou(context, prediction.split())
    labeled_text = insert_spans(context, bilou_tags)
    labeled_predictions.append(labeled_text)

# Save the labeled predictions to a CSV file
labeled_output_df = pd.DataFrame({
    'text': contexts,
    'causal_text_w_pairs': labeled_predictions
})
def evaluate_performance(true_labels, predicted_labels):
    # Flatten the label lists if they are in a sequence format
    true_labels_flat = [label for sublist in true_labels for label in sublist]
    predicted_labels_flat = [label for sublist in predicted_labels for label in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, predicted_labels_flat, average='weighted', zero_division=0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
labeled_output_df.to_csv('/teamspace/studios/this_studio/labeled_causal_text_w_pairs.csv', index=False)
print("Labeled predictions saved to /teamspace/studios/this_studio/labeled_causal_text_w_pairs.csv")
