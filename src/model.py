# =============================================================================
# Model 
# =============================================================================
from src import utils,metric
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-05
WEIGHT_DECAY = 1e-05
num_of_batches_per_epoch = len(X_train)//TRAIN_BATCH_SIZE


history = defaultdict(list)
class DistillBERT(torch.nn.Module):
    def __init__(self,num_classes):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.6)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        bert_last = hidden_state[:, 0]
        output = self.classifier(bert_last)
        return output    

class BERT(torch.nn.Module):
    def __init__(self,num_classes):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("bert-base-uncased",output_hidden_states=True)
        self.classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.6)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        bert_last = hidden_state[:, 0]
        output = self.classifier(bert_last)
        return output    

class RoBERTA(torch.nn.Module):
    def __init__(self,num_classes):
        super(BibBirdClass, self).__init__()
        self.l1 = BigBirdModel.from_pretrained("roberta-base",output_hidden_states=True)
        self.classifier = torch.nn.Linear(4096, 1024)
        self.dropout = torch.nn.Dropout(0.6)
        self.classifier = torch.nn.Linear(1024, num_classes)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        bert_last = hidden_state[:, 0]
        output = self.classifier(bert_last)
        return output                        


from transformers import DistilBertConfig,DistilBertTokenizer,DistilBertModel
from transformers import BertConfig,BertTokenizer,BertModel
from transformers import BigBirdConfig,BigBirdTokenizer,BigBirdModel
from transformers import LongformerConfig,LongformerTokenizer,LongformerModel
num_classes = len(df_profile.labels.unique())
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
baseline_model = BERTClass(num_classes)
baseline_model.to(device)

class BertDataFormat(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        cur_doc = str(self.data.doc[index])
        cur_doc = " ".join(cur_doc.split())
        inputs = self.tokenizer.encode_plus(
            cur_doc,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.labels[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

training_set = BertDataFormat(train_df, tokenizer, MAX_LEN)
testing_set = BertDataFormat(test_df, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

testing_set = BertDataFormat(test_df, tokenizer, MAX_LEN)
testing_loader = DataLoader(testing_set, **test_params)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  baseline_model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
