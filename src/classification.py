import os
import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import (
    SGDClassifier, 
    PassiveAggressiveClassifier
)

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    hamming_loss,
)
from scipy.stats import wilcoxon

from utils import (
    greek_stopwords, 
    lowercase_and_remove_accents, 
    rename_category_for_plot
)

def train_evaluate_classifiers(dataset_dir, vectorization_method):
    """
    Function that trains and evaluates the neural networks and 
    AdaBoost classifiers for the Greek_Reddit dataset, given a
    predefined vectorization method.
    """

    # Set 20 random seeds to be used for random weight initialization
    # during different training runs.
    random_seeds = [
        0, 1, 42, 5, 11, 10, 67, 45, 23, 20, 
        88, 92, 53, 31, 13, 2, 55, 3, 39, 72
    ]

    # Set the classification labels for the plots below.
    labels = ['κοινωνία', 'εκπαίδευση', 'ψυχαγωγία/κουλτούρα', 'πολιτική', 'τεχνολογία/επιστήμη', 'οικονομία', 'ταξίδια', 'αθλητικά', 'φαγητό', 'ιστορία']
    plot_labels = list(map(rename_category_for_plot, sorted(labels)))

    # Load the dataset splits.
    train_df = pandas.read_csv(os.path.join(dataset_dir, 'gr_reddit_train.csv'), index_col = False)
    test_df = pandas.read_csv(os.path.join(dataset_dir, 'gr_reddit_test.csv'), index_col = False)
    val_df = pandas.read_csv(os.path.join(dataset_dir, 'gr_reddit_val.csv'), index_col = False)

    # Separate the input features from the output labels.
    y_train = train_df.pop('category').to_numpy()
    X_train = train_df.pop('text')

    y_test = test_df.pop('category').to_numpy()
    X_test = test_df.pop('text')
    
    y_val = val_df.pop('category').to_numpy()
    X_val = val_df.pop('text')

    # Preprocess the dataset by lowercasing and removing accents.
    X_train = X_train.apply(lowercase_and_remove_accents)
    X_test = X_test.apply(lowercase_and_remove_accents)
    X_val = X_val.apply(lowercase_and_remove_accents)

    if vectorization_method == 'tfidf':
        # Initialize the tfidf vectorizer and fit it to the entire dataset.
        vectorizer = TfidfVectorizer(
            lowercase = True,
            use_idf = True,
            norm = None,
            sublinear_tf = True,
            stop_words = list(greek_stopwords),
            max_df = 0.99,
            min_df = 0.01,
            ngram_range = (1, 3)
        )
    
        # Transform the text columns using TF-IDF for each dataset split.
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        X_val = vectorizer.transform(X_val)
    else:
        # Define the mapping between the vectorization methods and their files.
        method_map = {
            'spacy': 'el_core_news_lg_embeddings.npy',
            'greekbert': 'bert-base-greek-uncased-v1_embeddings.npy',
            'mpnet': 'paraphrase-multilingual-mpnet-base-v2_embeddings.npy',
            'xlm-greek': 'stsb-xlm-r-greek-transfer_embeddings.npy'
        }

        # Select the vectorization file.
        vector_file = method_map[vectorization_method]

        # Load the vectorization file.
        X_train = numpy.load(os.path.join('embeddings', f'train_{vector_file}'), allow_pickle = True)
        X_test = numpy.load(os.path.join('embeddings', f'test_{vector_file}'), allow_pickle = True)
        X_val = numpy.load(os.path.join('embeddings', f'val_{vector_file}'), allow_pickle = True)

    # Reshape the category lists from 1D to 2D.
    y_train = numpy.reshape(y_train, (-1, 1))
    y_test = numpy.reshape(y_test, (-1, 1))
    y_val = numpy.reshape(y_val, (-1, 1))

    # Transform the output class labels using multi-hot encoding.
    binarizer = MultiLabelBinarizer()
    y_train = binarizer.fit_transform(y_train)
    y_test = binarizer.fit_transform(y_test)
    y_val = binarizer.fit_transform(y_val)

    # Get the classes from the binarizer.
    classes = binarizer.classes_
    print(classes)

    # Initialize the classifiers.
    mlpc = MLPClassifier(solver = 'lbfgs', max_iter = 1000, early_stopping = True)
    sgdc = SGDClassifier(penalty = 'l1', max_iter = 1000, early_stopping = True)
    pac = PassiveAggressiveClassifier(loss = 'squared_hinge', max_iter = 1000, early_stopping = True, shuffle = True)
    gbc = GradientBoostingClassifier(criterion = 'squared_error')
    
    evaluation_scores = {}
    # Set the number of threads for parallelization.
    n_jobs = -1 # Use all threads.
    
    for classifier in [mlpc, sgdc, pac, gbc]:

        # Get the classifier setup.
        if classifier.__class__.__name__ == 'MLPClassifier':
            classifier_setup = f'{classifier.__class__.__name__}: solver ({classifier.solver})'
        elif classifier.__class__.__name__ == 'SGDClassifier':
            classifier_setup = f'{classifier.__class__.__name__}: penalty ({classifier.penalty})'
        elif classifier.__class__.__name__ == 'PassiveAggressiveClassifier':
            classifier_setup = f'{classifier.__class__.__name__}: loss ({classifier.loss})'
        elif classifier.__class__.__name__ == 'GradientBoostingClassifier':
            classifier_setup = f'{classifier.__class__.__name__}: criterion ({classifier.criterion}):'

        print(classifier_setup, end = '\n\n')

        # Initializes the score lists.
        precision_scores, recall_scores, f1_scores, loss_scores = [], [], [], []

        # Repeat the experiment for each classifier 20 times.
        for _, random_seed in tqdm(zip(range(20), random_seeds), desc = 'Running experiment...'):

            # Set a random seed for the classifier, to ensure the results are reproducible.
            classifier.random_state = random_seed
            
            # Train a binary classifier for each class using the set amount of cpu threads.
            clf = OneVsRestClassifier(classifier, n_jobs = n_jobs)
            clf.fit(X_train, y_train)
       
            # Get all the predicted labels for each point in the test dataset.
            y_pred = clf.predict(X_test)

            # Calculate macro averaged metrics and the hamming loss for the classifier.
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average = 'macro', beta = 1.0)
            h_loss = hamming_loss(y_test, y_pred)

            # Save the metrics into lists.
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            loss_scores.append(h_loss)

            # Calculate the confusion matrix.
            cf_matrix = confusion_matrix(y_test.argmax(axis = 1), y_pred.argmax(axis = 1))

            # Create a pandas dataframe from the confusion matrix.
            df = pandas.DataFrame(cf_matrix, index = plot_labels, columns = plot_labels)

            # Plot the dataframe as a heatmap, and save it in a file.
            fig = plt.figure(figsize = (10, 7))
            seaborn.heatmap(df, cmap = 'Blues', annot = True, fmt = 'd')
            plot_path = f'images/cf_{f1}_{classifier.__class__.__name__}_{vectorization_method}.svg'
            plt.xticks(rotation = 35, ha = 'right')
            fig.savefig(plot_path, dpi = fig.dpi, format = 'svg', bbox_inches = 'tight')

        # Store scores for each classification setup.
        evaluation_scores[classifier_setup] = [
            precision_scores,
            recall_scores,
            f1_scores,
            loss_scores
        ]

    return evaluation_scores


def run_experiments(dataset_path, results_path):
        
    # Train and evaluate the classifiers using different vectorization methods.
    evaluation_scores = {
        'tfidf': {}, 'spacy': {}, 'greekbert': {}, 'mpnet': {}, 'xlm-greek': {}
    }
    
    for method in evaluation_scores.keys():
        evaluation_scores[method] = train_evaluate_classifiers(dataset_path, method)

    # Store the overall results.
    overall_results = [['Method', 'Precision', 'Recall', 'F1', 'Loss', 'p-value', 'statistics']]
    
    for method in evaluation_scores.keys():
        for classifier, scores in evaluation_scores[method].items():

            # Unpack the evaluation scores into variables.
            precision_scores, recall_scores, f1_scores, loss_scores = scores
            
            # Calculate the mean and std evaluation scores.
            precision_mean = round(numpy.mean(precision_scores) * 100, 2)
            recall_mean = round(numpy.mean(recall_scores) * 100, 2)
            f1_mean = round(numpy.mean(f1_scores) * 100, 2)
            loss_mean = round(numpy.mean(loss_scores) * 100, 2)
            
            precision_std = round(numpy.std(precision_scores) * 100, 2)
            recall_std = round(numpy.std(recall_scores) * 100, 2)
            f1_std = round(numpy.std(f1_scores) * 100, 2)
            loss_std = round(numpy.std(loss_scores) * 100, 2)

            # Calculate the Wilcoxon signed-rank test 
            # to confirm the statistical significance 
            # for the F1 score of each classifier 
            # over the baseline method (tfidf).
            statistics, p_value = 0, 0
            if method != 'tfidf':
                _, _, baseline_f1_scores, _ = evaluation_scores['tfidf'][classifier]
                statistics, p_value = wilcoxon(baseline_f1_scores, f1_scores)
            
            overall_results.append([
                f'{classifier} ({method})',
                f'{precision_mean} ± {precision_std}',
                f'{recall_mean} ± {recall_std}',
                f'{f1_mean} ± {f1_std}',
                f'{loss_mean} ± {loss_std}',
                f'{round(p_value, 4)}',
                f'{round(statistics, 4)}',
            ])

    # Join the list of results into a .csv format.
    csv_string = '\n'.join([','.join(l) for l in overall_results])

    # Save the string into a file.
    with open(results_path, 'w', encoding = 'utf-8') as csv_file:
        csv_file.write(csv_string)

    return

if __name__ == '__main__': run_experiments('Reddit/datasets', 'test_results.csv')
