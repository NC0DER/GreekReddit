from src.crawl import crawl_greek_subreddits
from src.process import *

def main():
    input_path = os.path.join('crawled_data', 'combined')
    csv_path = os.path.join('datasets', 'gr_reddit_preprocessed.csv')
    crawl_greek_subreddits()
    preprocess(input_path)
    postprocess(csv_path)
    analyze()
    return

if __name__ == '__main__': main()
