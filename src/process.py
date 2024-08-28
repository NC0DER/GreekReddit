import pathlib
import pandas
import numpy
import matplotlib.pyplot as plt

from src.utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def preprocess(input_path):
    """
    Function which preprocesses the csv files and merges them in a single csv file.
    """
    # Construct the input path.
    path = pathlib.Path(input_path)

    # Find all .csv files and their assosiated paths.
    csv_paths = [
        p.absolute()
        for p in path.iterdir()
        if p.is_file() and p.suffix == '.csv'
    ]

    # Append all dataframes to a list.
    df_list = []

    for csv_path in tqdm(csv_paths, desc = 'Preprocessing dataset...'):
        # Each dataframe contains exactly one row.
        row = pandas.read_csv(csv_path, index_col = False).iloc[0]

        # Remove html and urls.
        text = remove_html(row['text'])
        text = remove_urls(text)
        title = remove_urls(row['title'])

        # Incoporate user and email anonymization.
        text = anonymize_emails(text)
        text = anonymize_users(text)
        title = anonymize_emails(title)
        title = anonymize_users(title)

        # Replace emojis with space.
        text = replace_emoji_with_space(text)
        title = replace_emoji_with_space(title)

        # Replace non-breaking space with space.
        text = text.replace('​', ' ')

        # Remove extra whitespaces.
        text = ' '.join(text.split())
        title = ' '.join(title.split())

        # Create the new processed row as a dataframe of exactly one row.
        processed_row = pandas.DataFrame(
            zip([row['id']], [title], [text], [row['category']], [row['permalink']]),
                columns = ['id', 'title', 'text', 'category', 'url']
        )
        
        df_list.append(processed_row)

    # Make a new dataframe which combines the text from all processed dataframes.
    df = pandas.concat(df_list, axis = 0, join = 'outer')
    print('Preprocessed length: ', len(df))

    # Keep rows with text consisting of more than 25 words.
    df = df[df['text'].str.count(' ') > 25]
    print('Removed short posts length: ', len(df))

    # Remove rows with duplicate text.
    df = df.drop_duplicates(subset = 'text')
    print('Removed duplicate posts length: ', len(df))

    print(f'Overall dataset length after preprocessing: {len(df)}')

    # Save the processed dataframe into a .csv dataset.
    df.to_csv(os.path.join('datasets', 'gr_reddit_preprocessed.csv'), encoding = 'utf-8', index = False)

    return


def postprocess(filepath):
    """
    Function which postprocesses the dataset 
    and produces the train/val/test splits.
    """
    df = pandas.read_csv(filepath, index_col = False)

    # Join examples of highly similar classes.
    df['category'] = df['category'].apply(transform_category)
    
    # Save the processed dataframe into a .csv dataset.
    df.to_csv(f'gr_reddit_all.csv', encoding = 'utf-8', index = False)

    # Separate the input features from the output labels.
    y = df.pop('category')
    X = df

    # Generate train/validation/test split.
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify = y, test_size = 0.0764, shuffle = True, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.0834, shuffle = True, random_state = 42)

    # Join the input and the output features of each split.
    X_train['category'] = y_train
    X_val['category'] = y_val
    X_test['category'] = y_test

    # Save the processed dataframes into .csv datasets.
    X_train.to_csv(os.path.join('datasets', 'gr_reddit_train.csv'), encoding = 'utf-8', index = False)
    X_val.to_csv(os.path.join('datasets', 'gr_reddit_val.csv'), encoding = 'utf-8', index = False)
    X_test.to_csv(os.path.join('datasets', 'gr_reddit_test.csv'), encoding = 'utf-8', index = False)
    
    return


def analyze():
    """
    Function which analyzes the dataset.
    """
    datasets  = load_datasets()
    
    reddit_title_lengths = measure_token_lengths(datasets[0], 'title')
    reddit_title_sentence_counts = measure_sentence_counts(datasets[0], 'title')

    word_length_lists = []
    sentence_count_lists = []
    
    # Print the percentiles for the output lengths.
    for dataset in tqdm(datasets):
        text_lengths = measure_token_lengths(dataset, 'text')
        sentence_count = measure_sentence_counts(dataset, 'text')
        percentiles = [
            f'{p}%: {numpy.percentile(numpy.array(text_lengths), p)}' 
            for p in range(0, 125, 25)
        ]
        print(percentiles)
        word_length_lists.append(text_lengths)
        sentence_count_lists.append(sentence_count)

    labels = ['GreekReddit', 'Makedonia', 'OGTD', 'GLC', 'GreekSum']

    # Rename categories for class distribution plot.
    datasets[0]['category'] = datasets[0]['category'].apply(rename_category_for_plot)

    # Draw the boxplots and bar plot.
    plt.boxplot(word_length_lists, showfliers = False)
    plt.xticks([1, 2, 3, 4, 5], labels = labels)
    plt.savefig('images/boxplots.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.bar(datasets[0]['category'].value_counts().axes[0], datasets[0]['category'].value_counts().values)
    plt.xticks(rotation = 35, ha = 'right')
    plt.savefig('images/class_distribution.svg', format = 'svg', bbox_inches = 'tight')

    return
