# Automatic Identification of Character Actions
We describe a computational model to identify actions and its characters from scene descriptions found in movie scripts.  We frame this problem as a semantic-role labeling task, were the model has to label predicates and its constituents as actions, agents, patients or none.  This repo contains the code for the transformer-based models, and the data resources used to train these.


Pre-trained Models
---------------------

### SAIL access through Pier
Dataset and pre-trained models can be found in [Pier](https://pier.usc.edu:5001/index.cgi?launchApp=SYNO.SDS.App.FileStation3.Instance&launchParam=openfile%3D%252Fdata%252FmovieSRL%252F).

### External Access
To access the dataset and pre-trained models please fill [the form here](https://forms.gle/ZqJsPRMxDzHJ4YGD6).


Using the model
------------------------
The code required to run the model can be found in `movieSRL`. This code can be used with the pre-trained models or to train your own on new data.
To get you running, here are a few steps to follow.

First, import the tag set and create the translation directories. The tagset contains all the SRL tags, and the tag2idx gives us a numerical label for each of these tags
```
from alltags import *
tag2idx = {t:i for i, t in enumerate(tag_values)}
```
### Model specification

Next, we need to create a BERT configuration. This is where you specify things like the LSTM hidden_size, bidirectionality and the sequence output size (num. of output labels). Here, we extend on the BertConfig object from transformers
```
from transformers import BertConfig
config = BertConfig.from_pretrained('bert-base-cased')
config.num_labels = len(tag2idx)
config.hidden_size_lstm = FLAGS.hidden_size
config.bidirectional_lstm = FLAGS.bidirectional
```

### With a pre-trained model

After downloading the pre-trained model, unzip it into the MODEL_DIR directory (e.g., something like $HOME/movieSRL-gru50-bilateral).
Load the model's weights into a [SimpleBERTLSTM](https://github.com/usc-sail/mica-actions-agents-and-patients/blob/fdc9a4e53abfc0bc5e62e48e59eb030d6be7a4d6/movieSRL/models.py#L9) instance
```
from models import SimpleBERTLSTM
model = SimpleBERTLSTM.from_pretrained(MODEL_DIR, config = config)
model.eval()
```

### Input preprocessing
Input processing follows the traditional transformers pipeline. Sentences are tokenized using a pre-trained `BertTokenizer` (bert-base-cased), transformed into tensors, which the model takes as input.
```
import torch
import numpy as np
from transformers import BertTokenizer

sent = "Luke, I am your father!"

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokens = torch.tensor([tokenizer.encode(sent)])

with torch.no_grad():
  outputs = model(tokens)

```

### Post-processing
Outputs consist of posterior probabilities over the set of SRL tags. We need to convert it back to a human-readable format.

```
logits = outputs[0]
predictions = [list(p) for p in np.argmax(logits, axis = 2)]
```
