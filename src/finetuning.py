import os
import sys
import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import (
    load_models, 
    lowercase_and_remove_accents,
    rename_category_for_plot
)
from datasets import load_dataset
from typing import TypeVar, Dict
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer,
    AutoTokenizer,
    pipeline
)
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    confusion_matrix,
    precision_recall_fscore_support
)

# Generic type class for model and dataset objects.
Model = TypeVar('Model')
HFDataset = TypeVar('HFDataset')


def train_model(train_dataset: HFDataset, val_dataset: HFDataset, 
                tokenizer: Model, model: Model, output_dir: str, 
                label2id: Dict, learning_rate: float, train_batch_size: int,
                random_seed: int) -> None:
    """
    Parameters
    -----------
    train_dataset: HuggingFace training dataset (HFDataset).
    val_dataset: HuggingFace validation dataset (HFDataset).
    tokenizer: huggingface tokenizer model (Model).
    model: huggingface language model (Model).
    output_dir: directory to save model checkpoints (str).
    label2id: mapping dictionary from labels to ids (dict). 
    learning_rate: the learning rate value (float).
    train_batch_size: the batch size value used for training (int).
    random_seed: the random seed to be used for training (int).

    Returns
    --------
    None.
    """

    def preprocess_function(examples):

        # Lowercase the text and remove its accents.
        text_examples = list(map(lowercase_and_remove_accents, examples['text']))

        # Tokenize the text.
        model_inputs = tokenizer(
            text = text_examples,
            max_length = 512, 
            truncation = True,
            padding = True,
            return_tensors = 'pt'
        )
        
        # Assign an integer for each text label.
        model_inputs['label'] = [label2id[label] for label in examples['label']]
        return model_inputs


    # Apply the tokenization function to the datasets. 
    tokenized_train_dataset = train_dataset.map(
        preprocess_function, 
        batched = True,
        remove_columns = ['text']
    )

    tokenized_val_dataset = val_dataset.map(
        preprocess_function, 
        batched = True,
        remove_columns = ['text']
    )

    # As said in the documentation: "It's more efficient to dynamically pad 
    # the sentences to the longest length in a batch during collation, 
    # instead of padding the whole dataset to the maximum length."
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    # Define the evaluation metrics during fine-tuning.
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = 'macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Create a Hugginface Trainer object. 
    # When an existing model checkpoint is passed,
    # we are essentialy doing fine-tune.
    # If the execution is stopped, 
    # the latest checkpoint and its training state are automatically loaded.
    trainer = Trainer(
        model = model,
        args = TrainingArguments(
                output_dir = output_dir,
                learning_rate = learning_rate,
                per_device_train_batch_size = train_batch_size,
                per_device_eval_batch_size = 12,
                fp16 = False,
                ignore_data_skip = False,
                overwrite_output_dir = False,
                log_level = 'info',
                logging_steps = 100,
                evaluation_strategy = 'epoch',
                save_strategy = 'epoch',
                save_total_limit = 6,
                num_train_epochs = 5,
                resume_from_checkpoint = False,
                seed = random_seed
        ),
        train_dataset = tokenized_train_dataset,
        eval_dataset = tokenized_val_dataset,
        data_collator = data_collator,
        compute_metrics = compute_metrics
    )

    # Start the training loop.
    trainer.train(resume_from_checkpoint = False)

    return


def model_training():
        
    # Set the path for the model and greek reddit dataset.
    dataset_dir = 'Reddit/datasets'
    model_path = 'nlpaueb/bert-base-greek-uncased-v1'
    model_local_path = 'models/greekbert_reddit'
    
    # Set device.
    device = 'cuda:0'

    # Set 20 random seeds to be used for random weight initialization
    # during different training runs.
    random_seeds = [
        0, 1, 42, 5, 11, 10, 67, 45, 23, 20, 
        88, 92, 53, 31, 13, 2, 55, 3, 39, 72
    ]

    # Set the classification labels and their mappings.
    labels = ['κοινωνία', 'εκπαίδευση', 'ψυχαγωγία/κουλτούρα', 'πολιτική', 'τεχνολογία/επιστήμη', 'οικονομία', 'ταξίδια', 'αθλητικά', 'φαγητό', 'ιστορία']
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    # Load the datasets from the local path.
    try:
        train_dataset = load_dataset(
            'csv', 
            data_files = os.path.join(
                dataset_dir, 'gr_reddit_train.csv'), 
                split = 'all'
            )
        val_dataset = load_dataset(
            'csv', 
            data_files = os.path.join(
                dataset_dir, 'gr_reddit_val.csv'), 
                split = 'all'
            )
    except Exception as err:
        print(err, file = sys.stderr)
        raise FileNotFoundError(err)

    # Remove unnecessary columns and rename output columns.
    train_dataset = train_dataset.remove_columns(['id', 'title', 'url'])
    train_dataset = train_dataset.rename_column('category', 'label')

    val_dataset = val_dataset.remove_columns(['id', 'title', 'url'])
    val_dataset = val_dataset.rename_column('category', 'label')

    # Train, validate and save the model 20 times.
    for i, random_seed in tqdm(zip(range(20), random_seeds), desc = 'Training the model 20 times...'):

        # Load the required model and its tokenizer every time to train from scratch.
        tokenizer, model = load_models(
            language_model = model_path, 
            device = device,
            max_length = 512,
            num_labels = len(labels),
            id2label = id2label,
            label2id = label2id,
            dropout_rate = 0.1
        )

        train_model(train_dataset, 
            val_dataset,
            tokenizer,
            model,
            f'{model_local_path}_{str(i)}',
            label2id,
            learning_rate = 1e-05,
            train_batch_size = 16,
            random_seed = random_seed
        )

        # Save the tokenizer for each model.
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(os.path.join(f'{model_local_path}_{str(i)}', 'tokenizer'))
    
    return


def run_experiments(dataset_dir, model_local_path, results_path):

    # Set the classification labels and their mappings.
    labels = ['κοινωνία', 'εκπαίδευση', 'ψυχαγωγία/κουλτούρα', 'πολιτική', 'τεχνολογία/επιστήμη', 'οικονομία', 'ταξίδια', 'αθλητικά', 'φαγητό', 'ιστορία']
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    # Set the classification labels for the plots below.
    plot_labels = list(map(rename_category_for_plot, sorted(labels)))

    # Set device.
    device = 'cpu'

    # Load the test dataset from the local path.
    try:
        test_dataset = load_dataset(
            'csv', 
            data_files = os.path.join(dataset_dir, 'gr_reddit_test.csv'), 
                split = 'all'
        )
    except Exception as err:
        print(err, file = sys.stderr)
        raise FileNotFoundError(err)
    
    # Remove unnecessary columns and rename output columns.
    test_dataset = test_dataset.remove_columns(['id', 'title', 'url'])
    test_dataset = test_dataset.rename_column('category', 'label')

    # Initializes the score lists.
    precision_scores, recall_scores, f1_scores, loss_scores = [], [], [], []

    # Evaluate the 20 fine-tuned models.
    for i in tqdm(range(20), desc = 'Evaluate the 20 fine-tuned models...'):

        # Load the required model and its tokenizer every time for evaluation.
        tokenizer, model = load_models(
            language_model = f'{model_local_path}_{str(i)}', 
            device = device,
            max_length = 512,
            num_labels = len(labels),
            id2label = id2label,
            label2id = label2id,
            dropout_rate = 0.1,
            local_files_only = True
        )

        # Initialize the text classification pipeline.
        classifier = pipeline(
            task = 'text-classification', 
            model = model, 
            tokenizer = tokenizer,
            truncation = True,
            max_length = 512
        )

        # Predict the labels for the texts of the test dataset and evaluate them using the true labels.
        y_pred = [result['label'] for result in classifier(test_dataset['text'])]
        y_test = test_dataset['label']

        # Calculate macro averaged metrics and the hamming loss for the classifier.
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average = 'macro', beta = 1.0)
        h_loss = hamming_loss(y_test, y_pred)

        # Save the metrics into lists.
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        loss_scores.append(h_loss)

        # Calculate the confusion matrix.
        cf_matrix = confusion_matrix(y_test, y_pred)

        # Create a pandas dataframe from the confusion matrix.
        df = pandas.DataFrame(cf_matrix, index = plot_labels, columns = plot_labels)

        # Plot the dataframe as a heatmap, and save it in a file.
        fig = plt.figure(figsize = (10, 7))
        seaborn.heatmap(df, cmap = 'Blues', annot = True, fmt = 'd')
        plot_path = f'images/cf_{f1}_GreekBERT_{str(i)}.svg'
        plt.xticks(rotation = 35, ha = 'right')
        fig.savefig(plot_path, dpi = fig.dpi, format = 'svg', bbox_inches = 'tight')

    # Calculate the mean and std evaluation scores.
    precision_mean = round(numpy.mean(precision_scores) * 100, 2)
    recall_mean = round(numpy.mean(recall_scores) * 100, 2)
    f1_mean = round(numpy.mean(f1_scores) * 100, 2)
    loss_mean = round(numpy.mean(loss_scores) * 100, 2)
    
    precision_std = round(numpy.std(precision_scores) * 100, 2)
    recall_std = round(numpy.std(recall_scores) * 100, 2)
    f1_std = round(numpy.std(f1_scores) * 100, 2)
    loss_std = round(numpy.std(loss_scores) * 100, 2)
        
    # Store the overall results.
    overall_results = [['Method', 'Precision', 'Recall', 'F1', 'Loss']]

    overall_results.append([
        'GreekBERT',
        f'{precision_mean} ± {precision_std}',
        f'{recall_mean} ± {recall_std}',
        f'{f1_mean} ± {f1_std}',
        f'{loss_mean} ± {loss_std}'
    ])

    # Join the list of results into a .csv format.
    csv_string = '\n'.join([','.join(l) for l in overall_results])

    # Save the string into a file.
    with open(results_path, 'w', encoding = 'utf-8') as csv_file:
        csv_file.write(csv_string)

    return

if __name__ == '__main__': run_experiments(
    'datasets', 
    'models/epoch_4/greekbert_reddit',
    'greekbert_results_epoch_4.csv'
)
