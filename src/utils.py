import os
import re 
import html
import emoji
import warnings
import nltk
import pandas

from typing import TypeVar, Tuple
from statistics import mean
from datasets import load_dataset
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Generic type class for model objects.
Model = TypeVar('Model')

# Supress bs4 warning.
warnings.filterwarnings('ignore', category = MarkupResemblesLocatorWarning)


def detect_greek(model, text):
    """
    Function that detects if the given text is written in Greek,
    using the fasttext algorithm.
    """ 
    return model.predict(text, k = 1)[0][0] == '__label__el'  


def sent_tokenize(text):
    """
    Function that tokenizes texts into a list of sentences.
    """
    sentences = nltk.sent_tokenize(text, language = 'greek')
    sentences = (filter(None, re.split(r'(?<=[;;])\s+', sentence)) for sentence in sentences)
    return [sentence.strip() for sent_gen in sentences for sentence in sent_gen]


def word_tokenize(text):
    """
    Function that tokenizes texts into a list of words.
    """
    return nltk.RegexpTokenizer(r'[ ,;;.!?:-]+', gaps = True).tokenize(text)


def remove_html(text):
    """
    Function which strips HTML tags and unescapes HTML symbols from text.
    """
    return BeautifulSoup(html.unescape(text), features = 'html.parser').get_text(strip = True)


def remove_urls(text):
    """
    Function which removes URLs from text and removes the url formatting tags.
    """
    text = re.sub(r'\(*(http\S+|www\S+)', '', text)
    text = re.sub(r'[\[\]]', '', text)
    return text


def anonymize_emails(text):
    """
    Function which anonymizes email addresses from text.
    """
    return re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '<email>', text)


def replace_emoji_with_space(text):
    """
    Function which replaces emojis with space.
    """
    emoji_list = emoji.distinct_emoji_list(text)
    text = text.translate (
        str.maketrans(''.join(emoji_list), ' ' * sum(len(e) for e in (emoji_list)))
    )
    return text
    

def anonymize_users(text):
    """
    Function which anonymizes users.
    """
    text = re.sub(r'\/*u\/([^\s/]+)', '<user>', text)
    text = re.sub(r'@\w+', '<user>', text)
    return text


def transform_category(category):
    """
    Function that transforms the category of highly 
    similar classes into a common category.
    """
    if category == 'ψυχαγωγία' or category == 'κουλτούρα':
        category = 'ψυχαγωγία/κουλτούρα'
    elif category == 'τεχνολογία' or category == 'επιστήμη':
        category = 'τεχνολογία/επιστήμη'

    return category


def measure_token_lengths(df, label):
    """
    Function that measures and prints token lengths for a dataframe column.
    """

    text_lengths = [len(word_tokenize(text)) for text in df[label]]

    print(f'\n{label} - minimum token length: ', min(text_lengths))
    print(f'{label} - mean token length: ', mean(text_lengths))
    print(f'{label} - maximum token length: ', max(text_lengths))

    return text_lengths


def measure_sentence_counts(df, label):
    """
    Function that measures and prints sentence counts for a dataframe column.
    """

    sentence_counts = [len(sent_tokenize(text)) for text in df[label]]

    print(f'\n{label} - minimum sentence count: ', min(sentence_counts))
    print(f'{label} - mean sentence count: ', mean(sentence_counts))
    print(f'{label} - maximum sentence count: ', max(sentence_counts))

    return sentence_counts


def process_glc():
    """
    Function that processes the Greek Legal Codes dataset.
    It saves each split and also a combined version of the dataset to separate .csv files.
    It returns a list of the pandas dataframes. 
    """
    greek_legal_code = load_dataset('greek_legal_code')
    greek_legal_code.set_format(type = 'pandas')

    train = pandas.DataFrame(greek_legal_code['train'][:])
    test = pandas.DataFrame(greek_legal_code['test'][:])
    validation = pandas.DataFrame(greek_legal_code['validation'][:])
    combined = pandas.concat([train, test, validation])

    train.to_csv(os.path.join('datasets', 'greek_legal_code_train.csv'), encoding = 'utf-8', index = False)
    test.to_csv(os.path.join('datasets', 'greek_legal_code_test.csv'), encoding = 'utf-8', index = False)
    validation.to_csv(os.path.join('datasets', 'greek_legal_code_validation.csv'), encoding = 'utf-8', index = False)
    combined.to_csv(os.path.join('datasets', 'greek_legal_code.csv'), encoding = 'utf-8', index = False)
    
    return [combined, train, test, validation]


def process_greeksum():
    """
    Function that processes the GreekSum Classification dataset.
    It saves a combined version of the separate splits of the dataset to a .csv file. 
    """
    greeksum_train = pandas.read_csv(os.path.join('datasets', 'greeksum_classification_train.csv'), index_col = False)
    greeksum_test = pandas.read_csv(os.path.join('datasets', 'greeksum_classification_test.csv'), index_col = False)
    greeksum_valid = pandas.read_csv(os.path.join('datasets','greeksum_classification_valid.csv'), index_col = False)
    combined = pandas.concat([greeksum_train, greeksum_test, greeksum_valid])
    combined.to_csv(os.path.join('datasets', 'greeksum_classification.csv'), encoding = 'utf-8', index = False)

    return [combined, greeksum_train, greeksum_test, greeksum_valid]


def load_datasets():
    """
    Function that loads all the .csv datasets to a list of pandas dataframes.
    """
    reddit = pandas.read_csv(os.path.join('datasets', 'gr_reddit_all.csv'), index_col = False)
    makedonia = pandas.read_csv(os.path.join('datasets', 'makedonia.csv'), index_col = False)
    ogtd = pandas.read_csv(os.path.join('datasets', 'OGTDv1.csv'), index_col = False)
    glc = pandas.read_csv(os.path.join('datasets', 'greek_legal_code.csv'), index_col = False)
    greeksum = pandas.read_csv(os.path.join('datasets', 'greeksum_classification.csv'), index_col = False)

    return [reddit, makedonia, ogtd, glc, greeksum]


def load_reddit_dataset_splits():
    """
    Function that loads all the .csv files of the dataset splits to a list of pandas dataframes.
    """
    dataset_splits = []
    dataset_splits.append(pandas.read_csv(os.path.join('datasets','gr_reddit_train.csv'), index_col = False))
    dataset_splits.append(pandas.read_csv(os.path.join('datasets','gr_reddit_test.csv'), index_col = False))
    dataset_splits.append(pandas.read_csv(os.path.join('datasets','gr_reddit_val.csv'), index_col = False))

    return dataset_splits


def remove_greek_accents(text):
    """
    Function which replaces all lowercase greek accented characters
    with non-accented ones in text.
    """
    return text.translate(str.maketrans('άέόώήύϋΰίϊΐ', 'αεοωηυυυιιι'))


def lowercase_and_remove_accents(text):
    """
    Function which lowercases a greek text and removes its accents.
    """
    return remove_greek_accents(text.lower())


greek_stopwords = set(
    remove_greek_accents("""
    αδιάκοπα αι ακόμα ακόμη ακριβώς άλλα αλλά αλλαχού άλλες άλλη άλλην
    άλλης αλλιώς αλλιώτικα άλλο άλλοι αλλοιώς αλλοιώτικα άλλον άλλος άλλοτε αλλού
    άλλους άλλων άμα άμεσα αμέσως αν ανά ανάμεσα αναμεταξύ άνευ αντί αντίπερα αντίς
    άνω ανωτέρω άξαφνα απ απέναντι από απόψε άρα άραγε αρκετά αρκετές
    αρχικά ας αύριο αυτά αυτές αυτή αυτήν αυτής αυτό αυτοί αυτόν αυτός αυτού αυτούς
    αυτών αφότου αφού

    βέβαια βεβαιότατα

    γι για γιατί γρήγορα γύρω

    δα δε δείνα δεν δεξιά δήθεν δηλαδή δι δια διαρκώς δικά δικό δικοί δικός δικού
    δικούς διόλου δίπλα δίχως

    εάν εαυτό εαυτόν εαυτού εαυτούς εαυτών έγκαιρα εγκαίρως εγώ εδώ ειδεμή είθε είμαι
    είμαστε είναι εις είσαι είσαστε είστε είτε είχα είχαμε είχαν είχατε είχε είχες έκαστα
    έκαστες έκαστη έκαστην έκαστης έκαστο έκαστοι έκαστον έκαστος εκάστου εκάστους εκάστων
    εκεί εκείνα εκείνες εκείνη εκείνην εκείνης εκείνο εκείνοι εκείνον εκείνος εκείνου
    εκείνους εκείνων εκτός εμάς εμείς εμένα εμπρός εν ένα έναν ένας ενός εντελώς εντός
    εναντίον  εξής  εξαιτίας  επιπλέον επόμενη εντωμεταξύ ενώ εξ έξαφνα εξήσ εξίσου έξω επάνω
    επειδή έπειτα επί επίσης επομένως εσάς εσείς εσένα έστω εσύ ετέρα ετέραι ετέρας έτερες
    έτερη έτερης έτερο έτεροι έτερον έτερος ετέρου έτερους ετέρων ετούτα ετούτες ετούτη ετούτην
    ετούτης ετούτο ετούτοι ετούτον ετούτος ετούτου ετούτους ετούτων έτσι εύγε ευθύς ευτυχώς εφεξής
    έχει έχεις έχετε έχομε έχουμε έχουν εχτές έχω έως έγιναν  έγινε  έκανε  έξι  έχοντας

    η ήδη ήμασταν ήμαστε ήμουν ήσασταν ήσαστε ήσουν ήταν ήτανε ήτοι ήττον

    θα

    ι ιδία ίδια ίδιαν ιδίας ίδιες ίδιο ίδιοι ίδιον ίδιοσ ίδιος ιδίου ίδιους ίδιων ιδίως ιι ιιι
    ίσαμε ίσια ίσως

    κάθε καθεμία καθεμίας καθένα καθένας καθενός καθετί καθόλου καθώς και κακά κακώς καλά
    καλώς καμία καμίαν καμίας κάμποσα κάμποσες κάμποση κάμποσην κάμποσης κάμποσο κάμποσοι
    κάμποσον κάμποσος κάμποσου κάμποσους κάμποσων κανείς κάνεν κανένα κανέναν κανένας
    κανενός κάποια κάποιαν κάποιας κάποιες κάποιο κάποιοι κάποιον κάποιος κάποιου κάποιους
    κάποιων κάποτε κάπου κάπως κατ κατά κάτι κατιτί κατόπιν κάτω κιόλας κλπ κοντά κτλ κυρίως

    λιγάκι λίγο λιγότερο λόγω λοιπά λοιπόν

    μα μαζί μακάρι μακρυά μάλιστα μάλλον μας με μεθαύριο μείον μέλει μέλλεται μεμιάς μεν
    μερικά μερικές μερικοί μερικούς μερικών μέσα μετ μετά μεταξύ μέχρι μη μήδε μην μήπως
    μήτε μια μιαν μιας μόλις μολονότι μονάχα μόνες μόνη μόνην μόνης μόνο μόνοι μονομιάς
    μόνος μόνου μόνους μόνων μου μπορεί μπορούν μπρος μέσω  μία  μεσώ

    να ναι νωρίς

    ξανά ξαφνικά

    ο οι όλα όλες όλη όλην όλης όλο ολόγυρα όλοι όλον ολονέν όλος ολότελα όλου όλους όλων
    όλως ολωσδιόλου όμως όποια οποιαδήποτε οποίαν οποιανδήποτε οποίας οποίος οποιασδήποτε οποιδήποτε
    όποιες οποιεσδήποτε όποιο οποιοδηήποτε όποιοι όποιον οποιονδήποτε όποιος οποιοσδήποτε
    οποίου οποιουδήποτε οποίους οποιουσδήποτε οποίων οποιωνδήποτε όποτε οποτεδήποτε όπου
    οπουδήποτε όπως ορισμένα ορισμένες ορισμένων ορισμένως όσα οσαδήποτε όσες οσεσδήποτε
    όση οσηδήποτε όσην οσηνδήποτε όσης οσησδήποτε όσο οσοδήποτε όσοι οσοιδήποτε όσον οσονδήποτε
    όσος οσοσδήποτε όσου οσουδήποτε όσους οσουσδήποτε όσων οσωνδήποτε όταν ότι οτιδήποτε
    ότου ου ουδέ ούτε όχι οποία  οποίες  οποίο  οποίοι  οπότε  ος

    πάνω  παρά  περί  πολλά  πολλές  πολλοί  πολλούς  που  πρώτα  πρώτες  πρώτη  πρώτο  πρώτος  πως
    πάλι πάντα πάντοτε παντού πάντως πάρα πέρα πέρι περίπου περισσότερο πέρσι πέρυσι πια πιθανόν
    πιο πίσω πλάι πλέον πλην ποιά ποιάν ποιάς ποιές ποιό ποιοί ποιόν ποιός ποιού ποιούς
    ποιών πολύ πόσες πόση πόσην πόσης πόσοι πόσος πόσους πότε ποτέ πού πούθε πουθενά πρέπει
    πριν προ προκειμένου πρόκειται πρόπερσι προς προτού προχθές προχτές πρωτύτερα πώς

    σαν σας σε σεις σου στα στη στην στης στις στο στον στου στους στων συγχρόνως
    συν συνάμα συνεπώς συχνάς συχνές συχνή συχνήν συχνής συχνό συχνοί συχνόν
    συχνός συχνού συχνούς συχνών συχνώς σχεδόν

    τα τάδε ταύτα ταύτες ταύτη ταύτην ταύτης ταύτοταύτον ταύτος ταύτου ταύτων τάχα τάχατε
    τελευταία  τελευταίο  τελευταίος  τού  τρία  τρίτη  τρεις τελικά τελικώς τες τέτοια τέτοιαν
    τέτοιας τέτοιες τέτοιο τέτοιοι τέτοιον τέτοιος τέτοιου
    τέτοιους τέτοιων τη την της τι τίποτα τίποτε τις το τοι τον τοσ τόσα τόσες τόση τόσην
    τόσης τόσο τόσοι τόσον τόσος τόσου τόσους τόσων τότε του τουλάχιστο τουλάχιστον τους τούς τούτα
    τούτες τούτη τούτην τούτης τούτο τούτοι τούτοις τούτον τούτος τούτου τούτους τούτων τυχόν
    των τώρα

    υπ υπέρ υπό υπόψη υπόψιν ύστερα

    χωρίς χωριστά

    ω ως ωσάν ωσότου ώσπου ώστε ωστόσο ωχ
    """).split()
)


def spacy_useful_token(token):
    """
    Keep useful tokens which have 
       - Part Of Speech tag (POS): ['NOUN','PROPN','ADJ']
       - Alpha(token is word): True
       - Stop words(is, the, at, ...): False
    """
    return token.pos_ in ['NOUN','PROPN','ADJ'] and token.is_alpha and not token.is_stop and token.has_vector 


def load_models(
        language_model: str,
        device: str,
        max_length: int,
        num_labels: int,
        id2label: dict,
        label2id: dict,
        dropout_rate: float,
        local_files_only = False,
    ) -> Tuple[Model, Model]:
    """
    Utility function which loads the required models.
    
    Parameters
    ------------

    seq_classification_model: path to hugginface model (str).
    device: device to load and run model ['cpu', 'cuda:0'] (str).
    max_length: number of maximum input tokens (int).
    num_labels: number of labels for training (int).
    id2label: mapping dictionary from ids to labels - training only (dict).
    label2id: mapping dictionary from labels to ids - training only (dict).
    dropout_rate: the dropout rate for the model (float).
    local_files_only: boolean flag which loads only a local model (bool).

    Returns
    --------
    <object>: All model objects (Tuple[Model, Model]).
    """

    # Load the tokenizer either from the HuggingFace model hub or locally.
    if local_files_only:
        tokenizer = AutoTokenizer.from_pretrained(f'{language_model}/tokenizer/')
    else:
         tokenizer = AutoTokenizer.from_pretrained(language_model)

    # Load the pre-trained sequence classification model either from the HuggingFace model hub or locally.
    seq_classification_model = AutoModelForSequenceClassification.from_pretrained(
        language_model,
        problem_type = 'single_label_classification',
        max_length = max_length,
        num_labels = num_labels,
        id2label = id2label,
        label2id = label2id,
        local_files_only = local_files_only,
        attention_probs_dropout_prob = dropout_rate,
        hidden_dropout_prob = dropout_rate,
    )
    
    # Send the sequence classification model to the pre-specified device (cpu / gpu).
    seq_classification_model = seq_classification_model.to(device)

    return (tokenizer, seq_classification_model)


def rename_category_for_plot(category):
    """
    Function that renames the category for the class distribution plot.
    """
    match category:
        case 'κοινωνία': return 'κοινωνία (society)'
        case 'εκπαίδευση': return 'εκπαίδευση (education)'
        case 'ψυχαγωγία/κουλτούρα': return 'ψυχαγωγία/κουλτούρα (entertainment/culture)'
        case 'πολιτική': return 'πολιτική (politics)'
        case 'τεχνολογία/επιστήμη': return 'τεχνολογία/επιστήμη (technology/science)'
        case 'οικονομία': return 'οικονομία (economy)'
        case 'ταξίδια': return 'ταξίδια (travel)'
        case 'αθλητικά': return 'αθλητικά (sports)'
        case 'φαγητό': return 'φαγητό (food)'
        case 'ιστορία': return 'ιστορία (history)'
