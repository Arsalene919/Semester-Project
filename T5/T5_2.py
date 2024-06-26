import os
import torch
from datasets import load_dataset
from transformers import  T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Clear cached memory
torch.cuda.empty_cache()


# Load your datasets efficiently
dataset = load_dataset('csv', data_files={
    'train': '/teamspace/studios/this_studio/train_subtask2.csv',
    'validation': '/teamspace/studios/this_studio/dev_subtask2.csv'
}, cache_dir='./cache')
test_dataset = load_dataset('csv', data_files={
    'test': '/teamspace/studios/this_studio/test_subtask2_text.csv'
}, cache_dir='./cache')


# Function to print GPU memory usage
def print_gpu_utilization():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3} GB")
# Example usage in the training loop


# Step 4: Causal Component Identification using T5 with Gradient Checkpointing
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
t5_model.gradient_checkpointing_enable()

# Ensure use_cache=False when using gradient checkpointing
t5_model.config.use_cache = False
def convert_to_bilou(context, labels):
    words = context.split()
    bilou_labels = ["O"] * len(words)

    for label in labels:
        label_words = label.split()
        for i in range(len(words) - len(label_words) + 1):
            if words[i:i + len(label_words)] == label_words:
                if len(label_words) == 1:
                    bilou_labels[i] = "U-" + label_words[0]
                else:
                    bilou_labels[i] = "B-" + label_words[0]
                    for j in range(1, len(label_words) - 1):
                        bilou_labels[i + j] = "I-" + label_words[j]
                    bilou_labels[i + len(label_words) - 1] = "L-" + label_words[-1]
    return bilou_labels


# Prepare data for T5
def preprocess_t5_function(examples):
    inputs = ["extract causality: " + text for text in examples['text']]
    model_inputs = t5_tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    # Ensure labels are strings and convert to BILOU format
    contexts = examples['text']
    raw_labels = examples['text_w_pairs']
    labels = [convert_to_bilou(context, [raw_label]) for context, raw_label in zip(contexts, raw_labels)]
    labels = [" ".join(label) for label in labels if label]  # Filter out empty labels
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

# Preprocess datasets
encoded_train_val_datasets = dataset.map(preprocess_t5_function, batched=True)
encoded_test_dataset = test_dataset.map(test_preprocess_t5_function, batched=True)

training_args = TrainingArguments(
    output_dir='./t5_results',
    evaluation_strategy='steps',  # Evaluate periodically
    eval_steps=500,               # Adjust evaluation steps
    learning_rate=3e-5,
    per_device_train_batch_size=4,  # Adjust batch size as needed
    per_device_eval_batch_size=4,   # Adjust batch size as needed
    num_train_epochs=8,
    weight_decay=0.01,
    gradient_accumulation_steps=4,  # Accumulate gradients
    fp16=True,  # Enable mixed precision training
    eval_accumulation_steps=10,        # Accumulate steps for evaluation
    no_cuda=False                      # Use CPU for evaluation
)


# Compute metrics
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

    exact_matches = [pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]
    accuracy = sum(exact_matches) / len(exact_matches) if len(exact_matches) > 0 else 0.0
    
    flattened_preds = [token for sublist in decoded_preds for token in sublist.split()]
    flattened_labels = [token for sublist in decoded_labels for token in sublist.split()]

    min_length = min(len(flattened_preds), len(flattened_labels))
    flattened_preds = flattened_preds[:min_length]
    flattened_labels = flattened_labels[:min_length]

    precision = precision_score(flattened_labels, flattened_preds, average='weighted')
    recall = recall_score(flattened_labels, flattened_preds, average='weighted')
    f1 = f1_score(flattened_labels, flattened_preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }



trainer = Trainer(
    model=t5_model,
    args=training_args,
    train_dataset=encoded_train_val_datasets['train'],
    eval_dataset=encoded_train_val_datasets['validation'],
    tokenizer=t5_tokenizer,
    compute_metrics=compute_t5_metrics,
)

# Add logging
training_args.logging_dir = './logs'
training_args.logging_steps = 10




# Train and Evaluate
trainer.train()
eval_results = trainer.evaluate()
print("T5 Evaluation Results:", eval_results)

def identify_causal_components(text):
    inputs = t5_tokenizer("extract causality: " + text, return_tensors="pt").to("cuda")
    outputs = t5_model.generate(**inputs, num_beams=4, early_stopping=True)
    decoded_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output if decoded_output else ""


t5_predictions = []
t5_labels = []

for example in encoded_test_dataset['test']:
    prediction = identify_causal_components(example['text'])
    label = example['causal_text_w_pairs']
    print(f"Prediction: {prediction}, Label: {label}")  # Debug: Print prediction and label
    
    t5_predictions.append(prediction)
    t5_labels.append(label)

# Check for None values before comparing
t5_predictions = [pred if pred is not None else "" for pred in t5_predictions]
t5_labels = [label if label is not None else "" for label in t5_labels]

exact_matches = [pred.strip() == label.strip() for pred, label in zip(t5_predictions, t5_labels)]
accuracy = sum(exact_matches) / len(exact_matches) if len(exact_matches) > 0 else 0.0
print(f"T5 - Test Accuracy: {accuracy}")

flattened_preds = [token for sublist in t5_predictions for token in sublist.split()]
flattened_labels = [token for sublist in t5_labels for token in sublist.split()]

print(f"Flattened Predictions Length: {len(flattened_preds)}")
print(f"Flattened Labels Length: {len(flattened_labels)}")

min_length = min(len(flattened_preds), len(flattened_labels))
flattened_preds = flattened_preds[:min_length]
flattened_labels = flattened_labels[:min_length]

print(f"Aligned Flattened Predictions Length: {len(flattened_preds)}")
print(f"Aligned Flattened Labels Length: {len(flattened_labels)}")

precision = precision_score(flattened_labels, flattened_preds, average='weighted')
recall = recall_score(flattened_labels, flattened_preds, average='weighted')
f1 = f1_score(flattened_labels, flattened_preds, average='weighted')

print(f"T5 - Test Precision: {precision}")
print(f"T5 - Test Recall: {recall}")
print(f"T5 - Test F1 Score: {f1}")

# Run inference on a few examples from the test set
for i, example in enumerate(test_dataset):
    if i >= 5:  # Check the first 5 examples
        break
    text = example['text']
    true_label = example['causal_text_w_pairs']
    
    # Get the model's prediction
    predicted_label = identify_causal_components(text)
    
    # Print the text, true label, and predicted label
    print(f"Text: {text}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print("-" * 50)



