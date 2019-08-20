"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
"""

import sys
from gensim.corpora import WikiCorpus

def tokenize(content, token_min_len = 1, token_max_len = 50, lower = False):
    #override original method in wikicorpus.py
    return [token.encode('utf8') for token in content.split() 
           if len(token) <= 50 and not token.startswith('_')]

def make_corpus(in_f, out_f, num_articles):
    """Convert Wikipedia xml dump file to text corpus"""
    output = open(out_f, 'w+')
    wiki = WikiCorpus(in_f, tokenizer_func = tokenize)
    
    i = 0
    for text in wiki.get_texts():
        output.write((bytes(' ', 'utf-8').join(text)).decode('utf-8') + '\n')
        i += 1
        if (i % 100 == 0):
            print('Processed ' + str(i) + ' articles')
        if (i >= num_articles):
            break
    output.close()
    print('Processing complete!')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python dump.py <wikipedia_dump_file> <processed_text_file> <num_articles>')
        sys.exit(1)
    in_f = sys.argv[1]
    out_f = sys.argv[2]
    num_articles = int(sys.argv[3])
    make_corpus(in_f, out_f, num_articles)