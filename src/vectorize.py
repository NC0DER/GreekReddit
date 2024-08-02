import os
import spacy 
import numpy as np

from tqdm import tqdm
from utils import *
from sentence_transformers import SentenceTransformer

def generate_embeddings():
    """ 
    Generate embeddings for the dataset using different models.
    """

    # List of word and sentece embedding models.
    model_names = [
        'el_core_news_lg',
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
        'lighteternal/stsb-xlm-r-greek-transfer',
        'nlpaueb/bert-base-greek-uncased-v1'
    ]

    # Load the Greek_Reddit dataset.
    datasets = load_reddit_dataset_splits()
    dataset_splits = ['train', 'test', 'val']
    
    # Generate embeddings for the dataset using each separate model.
    for model_name in tqdm(model_names):
        for dataset_split, dataset in tqdm(zip(dataset_splits, datasets)):
            if model_name == 'el_core_news_lg':

                # Load spacy model.
                nlp = spacy.load(model_name)

                # Create an empty list to store embeddings for each text in the dataset.
                doc_vectors = []
                
                # Iterate through the dataset and convert each text to a vector representation.
                for text_col in tqdm(dataset['text']):

                    # Construct the spacy document from the lowercased text.
                    doc = nlp(text_col.lower())

                    # Split the text into tokens, convert useful tokens to vectors 
                    # and add them to a list of vectors.
                    vector_list = [token.vector for token in doc if spacy_useful_token(token)]

                    # If the vector list is non-empty, calculate the mean vector.
                    if vector_list:
                        doc_vector = np.mean(vector_list, axis = 0)
                    else:
                        doc_vector = np.zeros(300)
                        
                    doc_vectors.append(doc_vector)
            
                # Convert list of embeddings to numpy array and save it in a separate file.
                doc_vectors = np.asarray(doc_vectors, dtype = object)

                # Save the embedding representations of the datasets' texts.
                np.save(os.path.join('embeddings', f'{dataset_split}_{model_name}_embeddings'), doc_vectors)

            else: 
                # Create an empty list to store embeddings for each text in the dataset.
                doc_vectors = []

                # Load sentence embedding models.
                model = SentenceTransformer(model_name_or_path = model_name, device = 'cpu')
                
                # Iterate through the dataset and convert each text to a vector representation.
                for text_col in tqdm(dataset['text']):

                    # Lowercase the text.
                    text = text_col.lower()

                    # In contrast to other models, GreekBERT has only non-accented words in its vocabulary.
                    if model_name == 'nlpaueb/bert-base-greek-uncased-v1':
                        text = remove_greek_accents(text)

                    # Split the text to a list of sentences.
                    sentences = sent_tokenize(text)

                    # Create a list of sentence embeddings.
                    sent_vectors = [model.encode(sentence) for sentence in sentences] 

                    # Get the mean embedding of the document from the sentence embedding list.
                    doc_vector = np.mean(sent_vectors, axis = 0)
                    doc_vectors.append(doc_vector)
                    
                # Convert list of embeddings to numpy array and save it in a separate file.
                doc_vectors = np.asarray(doc_vectors, dtype = object)

                # Save the embedding representations of the datasets' texts.
                np.save(os.path.join('embeddings', f'{dataset_split}_{model_name.split("/")[-1]}_embeddings'), doc_vectors)
    return

if __name__ == '__main__': generate_embeddings()
