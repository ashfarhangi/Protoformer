<div align="center">

# Protoformer

[![Conference](http://img.shields.io/badge/PAKDD-2020-4b44ce.svg)](https://link.springer.com/chapter/10.1007/978-3-031-05933-9_35)
</div>

## Getting Started
All source code used to generate the results and figures in the paper are in
the `src` folder. The data used in this study is provided in `data` and the result figures are in `visualization`. See the `README.md` files in each directory for a full description.

### Datasets

Arvix: 
![](https://github.com/0415070/Protoformer/blob/main/visualization/arvix100.png)
### Installation

1. You can download a copy of all the files in this repository by cloning the following repo:

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
### Prerequisites
You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `environment.yml`. We used `conda` virtual environments to manage the project dependencies in
isolation. Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).
Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create



## License
All source code is made available under a GPL-3.0 license. You can freely use and modify the code, without warranty, as long as you provide attribution to the authors (e.g., citation). See LICENSE.md for the full license text.

The manuscript text is not open source. The authors reserve the rights to the article content, which is currently submitted for publication in the JOURNAL NAME.

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