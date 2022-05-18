# =============================================================================
# Main file  
# =============================================================================
from src import utils
from data import dataloader
from src import metric,model,plot,trainer


def run():
    """Builds model, loads data, trains and evaluates"""
    model = Protoformer('twitter-uni')
    # DistilBERT(twitter-uni) BERT(imdb) RoBERTa(arxiv-10)
    model.load_data('twitter-uni')
    # twitter-uni, imdb, arxiv-10
    model.build()
    model.train()
    model.evaluate()

if __name__ == '__main__':
    run()    