<div align="center">

# Protoformer

[![Conference](http://img.shields.io/badge/PAKDD-2020-4b44ce.svg)](https://link.springer.com/chapter/10.1007/978-3-031-05933-9_35)

## Getting Started

Instructions on setting up your project locally or on a cloud platform. To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.

- PyTorch 1.4.1
- Nvidia GPU 
### Datasets

Arvix: 
![](https://github.com/0415070/Protoformer/blob/main/visualization/arvix100.png)
### Installation

1. Clone the repo

   ```
   git clone https://github.com/ashfarhangi/Protoformer.git
   ```

2. Install requirement packages

   ```
   pip install -r requirements.txt
   ```

3. Enter your Twitter API keys in:
To use the data properly, you need to use your own official Twitter API. Please replace the API_KEY with you own. as shown below:
    

   ```
   const API_KEY = 'ENTER YOUR API';
   ```

4. Run model.py after the dataset has been gathered  


## Citation   

Farhangi, Ashkan, Ning Sui, Nan Hua, Haiyan Bai, Arthur Huang, and Zhishan Guo. "Protoformer: Embedding Prototypes for Transformers." In Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022, Chengdu, China, May 16–19, 2022, Proceedings, Part I, pp. 447-458. 2022.

```
@inproceedings{farhangi2022protoformer,
  title={Protoformer: Embedding Prototypes for Transformers},
  author={Farhangi, Ashkan and Sui, Ning and Hua, Nan and Bai, Haiyan and Huang, Arthur and Guo, Zhishan},
  booktitle={Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022, Chengdu, China, May 16--19, 2022, Proceedings, Part I},
  pages={447--458},
  year={2022}
}
```