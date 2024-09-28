import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')
os.environ["WANDB_MODE"] = "dryrun"

import warnings

warnings.filterwarnings("ignore", message="Unable to register cuDNN factory")
warnings.filterwarnings("ignore", message="Unable to register cuFFT factory")
warnings.filterwarnings("ignore", message="Unable to register cuBLAS factory")
warnings.filterwarnings("ignore", message="TF-TRT Warning: Could not find TensorRT")
warnings.filterwarnings("ignore", category=FutureWarning, module='datasets.table')
warnings.filterwarnings("ignore", message="load_metric is deprecated and will be removed in the next major version of datasets")
warnings.filterwarnings("ignore", message="Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate")
warnings.filterwarnings("ignore", message="Detected kernel version 4.15.0, which is below the recommended minimum of 5.5.0")

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_metric
from argparse import ArgumentParser
from datasets import concatenate_datasets, DatasetDict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

import evaluate

metric=evaluate.load("seqeval")

label_list = ['O','O','O','B-NEAR','B-NEL','O','B-NEN','B-NEO','B-NEP','O','B-NETI','O','O','O','I-NEAR','I-NEL','O','I-NEN','I-NEO','I-NEP','O','I-NETI','O']

id_to_label = {i : label_list[i] for i in range(len(label_list))}
label_to_id = {label_list[i] : i for i in range(len(label_list))}

num_labels = len(label_list)

from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("cfilt/HiNER-original-xlm-roberta-large")
model = AutoModelForTokenClassification.from_pretrained("cfilt/HiNER-original-xlm-roberta-large")

model.to(device)

def tokenize_function(examples):
    return tokenizer(examples["tokens"], padding="max_length", truncation=True, is_split_into_words=True)

from datasets import DatasetDict, Dataset

def convert_conll_to_dataset(file_path):
    sentences = []
    labels = []
    ids=[]

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        sentence = []
        label = []
        id_=0

        for line in lines:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    ids.append(id_)
                    sentence = []
                    label = []
                    id_+=1
            else:
                parts = line.split('\t')
                token = parts[0]
                tag = parts[-1]
                sentence.append(token)
                if tag in label_list:
                    label.append(label_to_id[tag])
                else:
                    label.append(6)

    if sentence:
        sentences.append(sentence)
        labels.append(label)
        ids.append(id_)
    data = {"id":ids,"tokens": sentences, "ner_tags": labels}
    dataset = Dataset.from_dict(data)

    return dataset


def tokenize_adjust_labels(all_samples_per_split):
  tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], padding='max_length', max_length=512, is_split_into_words=True, truncation=True)

  total_adjusted_labels = []

  for k in range(0, len(tokenized_samples["input_ids"])):
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids(batch_index=k)
    existing_label_ids = all_samples_per_split["ner_tags"][k]
    i = -1
    adjusted_label_ids = []

    for word_idx in word_ids_list:
      if(word_idx is None):
        adjusted_label_ids.append(-100)
      elif(word_idx!=prev_wid):
        i = i + 1
        adjusted_label_ids.append(existing_label_ids[i])
        prev_wid = word_idx
      else:
        label_name = label_list[existing_label_ids[i]]
        adjusted_label_ids.append(existing_label_ids[i])

    total_adjusted_labels.append(adjusted_label_ids)

  tokenized_samples["labels"] = total_adjusted_labels
  return tokenized_samples

def compute_metrics(p):
    predictions, labels = p

    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():

    label_list = ['O','O','O','B-NEAR','B-NEL','O','B-NEN','B-NEO','B-NEP','O','B-NETI','O','O','O','I-NEAR','I-NEL','O','I-NEN','I-NEO','I-NEP','O','I-NETI','O']

    id_to_label = {i : label_list[i] for i in range(len(label_list))}
    label_to_id = {label_list[i] : i for i in range(len(label_list))}

    print(label_to_id)

    num_labels = len(label_list)


    parser = ArgumentParser()
    parser.add_argument('--train', dest='train', help='Enter the train file')
    parser.add_argument('--dev', dest='dev', help='Enter the dev file')
    parser.add_argument('--model', dest='model', help='Enter the model file name')
    args = parser.parse_args()

    print("train",args.train)
    print("dev",args.dev)

    dataset_train = convert_conll_to_dataset(args.train)
    dataset_test = convert_conll_to_dataset(args.dev)

    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Test dataset size: {len(dataset_test)}")

    concatenated_dataset = DatasetDict({
        "train": dataset_train,
        "validation": dataset_test,
    })
    
    train_dataset = concatenated_dataset["train"]
    train_dataset = train_dataset.map(
        tokenize_adjust_labels,
        batched=True,
        num_proc=32,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
    )
    test_dataset = concatenated_dataset["validation"]
    test_dataset = test_dataset.map(
        tokenize_adjust_labels,
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Running tokenizer on test dataset",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    metric = load_metric("seqeval")

    batch_size = 4

    logging_steps = len(concatenated_dataset['train'])

    epochs = 5
    print("training args to be executed")
    training_args = TrainingArguments(
        output_dir="/scratch/sankalp/out_dir",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        )
    print("training args defined")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    print("training started")

    trainer.train()

    trainer.evaluate()

    model_name=args.model

    trainer.save_model(model_name)

    file_name=model_name+"-stats.txt"

    with open(file_name,"w") as f:
        f.write("epochs: ")
        f.write(str(epochs))
        f.write("\nbatch size: ")
        f.write(str(batch_size))

if __name__ == '__main__':
    main()
