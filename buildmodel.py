from datasets import load_dataset, load_metric
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Train Model
# load dataset and get the dataset splits
from datasets import load_dataset, load_metric
dataset = load_dataset("surrey-nlp/PLOD-CW")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]  
# Take labels from strings to class indexes for use in data analysis
label_encoding = {"B-O": 0, "B-AC": 1, "B-LF": 2, "I-LF": 2}

train_label_list = []
for sample in train_dataset["ner_tags"]:
    train_label_list.append([label_encoding[tag] for tag in sample])

val_label_list = []
for sample in val_dataset["ner_tags"]:
    val_label_list.append([label_encoding[tag] for tag in sample])

test_label_list = []
for sample in test_dataset["ner_tags"]:
    test_label_list.append([label_encoding[tag] for tag in sample])

tokenizer2 = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model2 = AutoModelForTokenClassification.from_pretrained("sentence-transformers/all-mpnet-base-v2", num_labels=3)

#tokenized_input = tokenizer2(train_dataset["tokens"], is_split_into_words=True)
def tokenize_and_align_labels(train_dataset, list_name):
    tokenized_inputs = tokenizer2(train_dataset["tokens"], truncation=True, is_split_into_words=True) ## For some models, you may need to set max_length to approximately 500.

    labels = []
    for i, label in enumerate(list_name):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_train_datasets = tokenize_and_align_labels(train_dataset, train_label_list)
tokenized_val_datasets = tokenize_and_align_labels(val_dataset, val_label_list)
tokenized_test_datasets = tokenize_and_align_labels(test_dataset, test_label_list)

def turn_dict_to_list_of_dict(d):
    new_list = []

    for labels, inputs in zip(d["labels"], d["input_ids"]):
        entry = {"input_ids": inputs, "labels": labels}
        new_list.append(entry)

    return new_list

#use 
tokenised_train = turn_dict_to_list_of_dict(tokenized_train_datasets)
tokenised_val = turn_dict_to_list_of_dict(tokenized_val_datasets)
tokenised_test = turn_dict_to_list_of_dict(tokenized_test_datasets)

#setup data collator
data_collator = DataCollatorForTokenClassification(tokenizer2)

#load metric
metric = load_metric("seqeval",trust_remote_code=True)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [train_label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [train_label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Training arguments
model_name = "sentence-transformers/all-mpnet-base-v2"
#model_name = "thenlper/gte-base"
epochs = 12
batch_size = 4
learning_rate = 6e-5

args = TrainingArguments(
    f"all-mpnet-base-v2-finetuned-NER",
    #f"thenlper/gte-base-NER",
    # evaluation_strategy = "epoch", ## Instead of focusing on loss and accuracy, we will focus on the F1 score
    evaluation_strategy ='steps',
    eval_steps = 7000,
    save_total_limit = 3,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.001,
    save_steps=35000,
    metric_for_best_model = 'f1',
    load_best_model_at_end=True
)

trainer = Trainer(
    model2,
    args,
    train_dataset=tokenised_train,
    eval_dataset=tokenised_val,
    data_collator = data_collator,
    tokenizer=tokenizer2,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

# Prepare the test data for evaluation in the same format as the training data
predictions, labels, _ = trainer.predict(tokenised_test)
predictions = np.argmax(predictions, axis=2)

true_labels = []
true_predictions = []
true_labels_sep = []
true_predictions_sep = []
for i in range(len(labels)):
  templist = []#range(sum(labels[0] != -100))
  templist2 = []
  for j in range(len(labels[i])):
    if labels[i][j]!= -100:
      true_labels.append(labels[i][j])
      true_predictions.append(predictions[i][j])
      templist.append(labels[i][j])
      templist2.append(predictions[i][j])
  true_labels_sep.append(templist)
  true_predictions_sep.append(templist2)


conf_mat = confusion_matrix(true_labels, true_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap='Blues', xticklabels=["Other", "Abbreviation","Long Form"], yticklabels=["Other", "Abbreviation","Long Form"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Fine Tuning')
plt.savefig('confusionmatrix.png')

trainer.save_model()