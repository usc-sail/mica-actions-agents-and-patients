# MovieSRL: Automatic Identification of Character Actions
We describe a computational model to identify actions and its characters from scene descriptions found in movie scripts.  We frame this problem as a semantic-role labeling task, were the model has to label predicates and its constituents as actions, agents, patients or none.  This repo contains the code for the transformer-based models, and the data resources used to train these.


## Pretrained Models
---------------------

### Through Pier
Dataset and pre-trained models can be found in [Pier](https://pier.usc.edu:5001/index.cgi?launchApp=SYNO.SDS.App.FileStation3.Instance&launchParam=openfile%3D%252Fdata%252FmovieSRL%252F)

### Request Access
To access the dataset and pre-trained models please fill [the form here](https://forms.gle/ZqJsPRMxDzHJ4YGD6).


## Using the model
------------------------
Import the tag set and create the translation directories
```
from alltags import *

tag2idx = {t:i for i, t in enumerate(tag_values)}
```

Create a config object. This is where you specify things like the LSTM hidden_size, bidirectionality and the sequence output size (num. of output labels)
```
from transformers import BertConfig
config = BertConfig.from_pretrained(MODEL_DIR)
config.num_labels = len(tag2idx)
config.hidden_size = 50
config.bidirectional = True
```

Load the model's weights into the SimpleBERTLSTM object
```
from models import SimpleBERTLSTM
model = SimpleBERTLSTM.from_pretrained(MODEL_DIR, config = config)
model.eval()
```

### Input preprocessing

```
import torch
import numpy as np
from transformers import BertTokenizer


sent = "Luke, I am your father!"
```

```
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokens = torch.tensor([tokenizer.encode(sent)])
```

### Prediction
```
with torch.no_grad():
  outputs = model(tokens)
```
