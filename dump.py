"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html
"""

import sys
from gensim.corpora import WikiCorpus

def make_corpus(in_f, out_f, num_articles):
    """Convert Wikipedia xml dump file to text corpus"""
    output = open(out_f, 'w+')
    wiki = WikiCorpus(in_f)
    
    i = 0
    for vec in wiki:
#        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        print(vec)
        i += 1
        if (i % 100 == 0):
            print('Processed ' + str(i) + ' articles')
        if (i >= num_articles):
            break
    output.close()
    print('Processing complete!')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file> <num_articles>')
        sys.exit(1)
    in_f = sys.argv[1]
    out_f = sys.argv[2]
    num_articles = int(sys.argv[3])
    make_corpus(in_f, out_f, num_articles)