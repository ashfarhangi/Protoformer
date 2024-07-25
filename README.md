<div align="center">

# Protoformer: Embedding Prototypes for Transformers
[![Conference](http://img.shields.io/badge/PAKDD-2022-4b44ce.svg)](https://link.springer.com/chapter/10.1007/978-3-031-05933-9_35)
</div>

**Published in**: [Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022](https://pakdd2022.org)

## Overview

Transformers have been widely applied in text classification, but real-world data often contain anomalies and noisy labels that challenge their performance. Protoformer is a novel self-learning framework for Transformers that leverages problematic samples to improve text classification. The framework features a selection mechanism for embedding samples, allowing efficient extraction and utilization of anomaly prototypes and difficult class prototypes.

## Key Features

- **Anomaly Detection**: Automatically detects and adjusts noisy labels to make the model more robust to complex datasets.
- **Prototype Selection**: Utilizes a selection mechanism for embedding samples, focusing on both anomaly and difficult class prototypes.
- **Improved Performance**: Demonstrates enhanced performance on datasets with diverse textual structures (e.g., Twitter, IMDB, ArXiv).

## Contributions

1. **Novel Framework**: Leverages harder-to-classify and anomaly samples, providing a solution for classifying complex datasets from the Internet.
2. **Label Adjustment Procedure**: Robust to noise, making the framework suitable for noisy Internet data and promoting a more robust Transformer model.
3. **Empirical Evaluation**: Evaluated on multiple datasets with both clean and noisy labels, showing significant performance improvements.

## Getting Started
All source code used to generate the results and figures in the paper are in
the `src` folder. The data used in this study is provided in `data` and the result figures are in `visualization`. See the `README.md` files in each directory for a full description.

### Datasets
Protoformer has been tested on the following datasets:

Twitter Data: Historical tweets for text classification tasks.
IMDB Reviews: Movie reviews dataset for sentiment analysis.
ArXiv Papers: Abstracts and titles of scientific papers for classification tasks.

### Installation

• You can download a copy of all the files in this repository by cloning the repo:

   ```Python
   git clone https://github.com/ashfarhangi/Protoformer.git
   ```

• Install requirement packages

   ```Python
   pip install -r requirements.txt
   ```

• Run model.py 

(optional)
• Enter your Twitter API keys in:
To use the data properly, you need to use your own official Twitter API. Please replace the API_KEY with you own. as shown below:
    

   ```
   const API_KEY = 'ENTER YOUR API';
   ```


### Prerequisites
```
Python
git
pip
```


## License
All source code is made available under a GPL-3.0 license. You can freely use and modify the code, without warranty, as long as you provide attribution to the authors (e.g., citation). See LICENSE.md for the full license text.


## Citation   
```
@inproceedings{farhangiprotformer,
  title={Protoformer: Embedding Prototypes for Transformers},
  author={Farhangi, Ashkan and Sui, Ning and Hua, Nan and Bai, Haiyan and Huang, Arthur and Guo, Zhishan},
  booktitle={Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022},
  year={2022},
  organization={PAKDD}
}
```
