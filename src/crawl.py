import os
import praw
import timeit
import time
import pandas
import fasttext
import json

from tqdm import tqdm
from itertools import product
from src.process import detect_greek

# Load the fasttext language detection model once.
model = fasttext.load_model('models/lid.176.bin')

# Initiate Reddit object with your client_id, secret and user agent.
reddit = praw.Reddit(
    client_id = 'example_id',
    client_secret = 'example_secret',
    user_agent = 'r_greece by u/example_username'
)

def greece_crawl():
    """
    Function that crawls the r/greece subreddit's posts 
    querying specific categories and keywords for specific time intervals.
    """

    # Specify the target subreddit.
    subreddit = reddit.subreddit('greece')
    
    # Load the keywords for each category.
    categories = {}
    with open(os.path.join('keywords', 'keywords.json'), 'r', encoding = 'utf-8') as f: 
        categories = json.load(f)
    
    # Dictionary to map flairs to categories.
    flair_to_category = {
        ':zz_art: πολιτιστικά/culture': 'κουλτούρα',
        ':zz_travel: travel/τουρισμός': 'ταξίδια',
        ':zz_politics: πολιτική/politics': 'πολιτική',
        ':zz_society: κοινωνία/society': 'κοινωνία',
        ':zz_economy: οικονομία/economy': 'οικονομία',
        ':zz_science: επιστήμη/science': 'επιστήμη',
        ':zz_sports: αθλητισμός/sports': 'αθλητικά',
        ':zz_education: εκπαίδευση/education': 'εκπαίδευση',
        ':zz_history: ιστορία/history': 'ιστορία',
        ':zz_technology: τεχνολογία/technology':'τεχνολογία',
        ':zz_entertainment: ψυχαγωγία/entertainment': 'ψυχαγωγία',
        ':zz_food: κουζίνα/food': 'φαγητό'
    }

    # Iterate through the categories and their associated keywords, 
    # while selecting posts written in Greek for specific time intervals.
    for category, keywords in tqdm(categories.items()):
        for keyword in tqdm(keywords):
            for filter_sort, time_filter in product(['top', 'hot', 'relevance', 'comments', 'new'], ['all', 'year', 'month', 'week']): 
                start = timeit.default_timer()
                posts = list(subreddit.search(keyword, limit = None, sort = filter_sort, time_filter = time_filter))
                
                if time_filter == 'all' and filter_sort == 'top' and not posts: 
                    print(keyword)
                
                for post in posts:
                    if post.link_flair_text in flair_to_category.keys() and len(post.selftext.split()) > 25 and detect_greek(model, post.selftext.replace('\n', '')) and not post.over_18: 
                        
                        # Create a pandas dataframe for each subreddit post and save it to a csv.
                        post_df = pandas.DataFrame(zip([post.id], [post.title], [post.selftext], [flair_to_category[post.link_flair_text]], [f'https://www.reddit.com{post.permalink}']),
                        columns = ['id', 'title', 'text', 'category', 'permalink'])
                        post_df.to_csv(os.path.join('crawled_data', 'new', f'{flair_to_category[post.link_flair_text]}_{post.id}.csv'), encoding = 'utf-8', index = False)
                
                end = timeit.default_timer()
                processing_time = end - start

                # If the processing time is less than four seconds, sleep for the remaining duration.
                if processing_time < 4.0:
                    time.sleep(4.0 - processing_time)

    return


def thessaloniki_crawl():
    """
    Function that crawls the r/thessaloniki subreddit's posts 
    querying specific categories and keywords for specific time intervals.
    """

    # Specify the target subreddit.
    subreddit = reddit.subreddit('thessaloniki')
    
    # Load the keywords for each category.
    categories = {}
    with open(os.path.join('keywords', 'thessaloniki_keywords.json'), 'r', encoding = 'utf-8') as f: 
        categories = json.load(f)
    
    # Dictionairy to map flairs to categories.
    flair_to_category = {
        ":airplane: Travel / Ταξίδι": 'ταξίδια',
        ":cuisine: Cuisine / Κουζίνα": 'φαγητό',
        ":history: History / Ιστορία": 'ιστορία'
    }

    # Iterate through the categories and their associated keywords, 
    # while selecting posts written in Greek for specific time intervals.
    for category, keywords in tqdm(categories.items()):
        for keyword in tqdm(keywords):
            for filter_sort, time_filter in product(['top', 'hot', 'relevance', 'comments', 'new'], ['all', 'year', 'month', 'week']): #'top', 'hot', 'relevance', 'comments','new'  'all', 'year', 'month'
                start = timeit.default_timer()
                posts = list(subreddit.search(keyword, limit = None, sort = filter_sort, time_filter = time_filter))
                
                if time_filter == 'all' and filter_sort == 'top' and not posts: 
                    print(keyword)
                
                for post in posts:
                    if post.link_flair_text in flair_to_category.keys() and len(post.selftext.split()) > 25 and detect_greek(model, post.selftext.replace('\n', '')) and not post.over_18: 
                        
                        # Create a pandas dataframe for each subreddit post and save it to a csv.
                        post_df = pandas.DataFrame(zip([post.id], [post.title], [post.selftext], [flair_to_category[post.link_flair_text]], [f'https://www.reddit.com{post.permalink}']),
                        columns = ['id', 'title', 'text', 'category', 'permalink'])
                        post_df.to_csv(os.path.join('crawled_data', 'thessaloniki', f'{flair_to_category[post.link_flair_text]}_{post.id}.csv'), encoding = 'utf-8', index = False)
                
                end = timeit.default_timer()
                processing_time = end - start

                # If the processing time is less than four seconds, sleep for the remaining duration.
                if processing_time < 4.0:
                    time.sleep(4.0 - processing_time)

    return


def greek_sports_crawl():
    
    """
    Function that crawls subreddits related to greek sports teams and contain posts written in Greek.
    """

    # List of greek sports team subreddits
    subs = ['GreekFooty', 'OlympiakosFC', 'OlympiacosBC', 'PAOK', 'Panathinaikos', 'AEKAthensFC', 'AEKAthensBC', 'AEKAthensWFC' ]

    for sub in subs:
        posts = []
        subreddit = reddit.subreddit(sub)
        for time_filter in ['all', 'year', 'month','week']:
            start = timeit.default_timer()
            posts.extend(list(subreddit.top(time_filter = time_filter)))
            end = timeit.default_timer()
            processing_time = end - start

            # If the processing time is less than four seconds, sleep for the remaining duration.
            if processing_time < 4.0:
                time.sleep(4.0 - processing_time)

        start = timeit.default_timer()
        posts.extend(list(subreddit.new()))
        end = timeit.default_timer()
        processing_time = end - start
        # If the processing time is less than four seconds, sleep for the remaining duration.
        if processing_time < 4.0:
                time.sleep(4.0 - processing_time)

        start = timeit.default_timer()
        posts.extend(list(subreddit.hot()))
        end = timeit.default_timer()
        processing_time = end - start
        # If the processing time is less than four seconds, sleep for the remaining duration.
        if processing_time < 4.0:
                time.sleep(4.0 - processing_time)

        start = timeit.default_timer()
        posts.extend(list(subreddit.rising()))
        end = timeit.default_timer()
        processing_time = end - start
        # If the processing time is less than four seconds, sleep for the remaining duration.
        if processing_time < 4.0:
                time.sleep(4.0 - processing_time)
                    
        for post in posts:
            if len(post.selftext.split()) > 25 and detect_greek(model, post.selftext.replace('\n', '')) and not post.over_18: 
                
                # Create a pandas dataframe for each subreddit post and save it to a csv.
                post_df = pandas.DataFrame(zip([post.id], [post.title], [post.selftext], ['αθλητικά'], [f'https://www.reddit.com{post.permalink}']),
                columns = ['id', 'title', 'text', 'category', 'permalink'])
                post_df.to_csv(os.path.join('crawled_data', 'sports', f'αθλητικά_{post.id}.csv'), encoding = 'utf-8', index = False)
    return


def greek_food_crawl():
    
    """
    Function that crawls food related subreddits and contain posts written in Greek.
    """

    # List of greek football team subreddits
    subs = ['Greek_Food', 'greekfood']

    for sub in tqdm(subs):
        subreddit = reddit.subreddit(sub)
        
        posts = []
        for time_filter in ['all', 'year', 'month','week']:
            start = timeit.default_timer()
            posts.extend(list(subreddit.top(time_filter = time_filter)))
            end = timeit.default_timer()
            processing_time = end - start

            # If the processing time is less than four seconds, sleep for the remaining duration.
            if processing_time < 4.0:
                time.sleep(4.0 - processing_time)

        start = timeit.default_timer()
        posts.extend(list(subreddit.new()))
        end = timeit.default_timer()
        processing_time = end - start

        # If the processing time is less than four seconds, sleep for the remaining duration.
        if processing_time < 4.0:
                time.sleep(4.0 - processing_time)

        start = timeit.default_timer()
        posts.extend(list(subreddit.hot()))
        end = timeit.default_timer()
        processing_time = end - start

        # If the processing time is less than four seconds, sleep for the remaining duration.
        if processing_time < 4.0:
                time.sleep(4.0 - processing_time)

        start = timeit.default_timer()
        posts.extend(list(subreddit.rising()))
        end = timeit.default_timer()
        processing_time = end - start

        # If the processing time is less than four seconds, sleep for the remaining duration.
        if processing_time < 4.0:
                time.sleep(4.0 - processing_time)
                    
        for post in posts:
            if len(post.selftext.split()) > 25 and detect_greek(model, post.selftext.replace('\n', '')) and not post.over_18: 
                
                # Create a pandas dataframe for each subreddit post and save it to a csv.
                post_df = pandas.DataFrame(zip([post.id], [post.title], [post.selftext], ['φαγητό'], [f'https://www.reddit.com{post.permalink}']),
                columns = ['id', 'title', 'text', 'category', 'permalink'])
                post_df.to_csv(os.path.join('crawled_data' ,'greek_food', f'φαγητό_{post.id}.csv'), encoding = 'utf-8', index = False)
    
    return


def personal_finance_crawl():
    """
    Function that crawls the r/PersonalFinanceGreece subreddit.
    """

    subreddit = reddit.subreddit('PersonalFinanceGreece')
    
    posts = []
    for time_filter in ['all', 'year', 'month','week']:
        start = timeit.default_timer()
        posts.extend(list(subreddit.top(time_filter = time_filter)))
        end = timeit.default_timer()
        processing_time = end - start

        # If the processing time is less than four seconds, sleep for the remaining duration.
        if processing_time < 4.0:
            time.sleep(4.0 - processing_time)

    start = timeit.default_timer()
    posts.extend(list(subreddit.new()))
    end = timeit.default_timer()
    processing_time = end - start

    # If the processing time is less than four seconds, sleep for the remaining duration.
    if processing_time < 4.0:
            time.sleep(4.0 - processing_time)

    start = timeit.default_timer()
    posts.extend(list(subreddit.hot()))
    end = timeit.default_timer()
    processing_time = end - start

    # If the processing time is less than four seconds, sleep for the remaining duration.
    if processing_time < 4.0:
            time.sleep(4.0 - processing_time)

    start = timeit.default_timer()
    posts.extend(list(subreddit.rising()))
    end = timeit.default_timer()
    processing_time = end - start

    # If the processing time is less than four seconds, sleep for the remaining duration.
    if processing_time < 4.0:
            time.sleep(4.0 - processing_time)
                
    for post in posts:
        if len(post.selftext.split()) > 25 and detect_greek(model, post.selftext.replace('\n', '')) and not post.over_18: 
            
            # Create a pandas dataframe for each subreddit post and save it to a csv.
            post_df = pandas.DataFrame(zip([post.id], [post.title], [post.selftext], ['οικονομικά'], [f'https://www.reddit.com{post.permalink}']),
            columns = ['id', 'title', 'text', 'category', 'permalink'])
            post_df.to_csv(os.path.join('crawled_data' ,'PersonalFinanceGreece', f'οικονομικά_{post.id}.csv'), encoding = 'utf-8', index = False)

    return


def crawl_greek_subreddits():
    """
    Function that crawls various greek subreddits sequencially.
    """

    greece_crawl()
    thessaloniki_crawl()
    greek_sports_crawl()
    greek_food_crawl()
    personal_finance_crawl()

    return
