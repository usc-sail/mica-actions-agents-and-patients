# Code from https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
import torch
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

import sys
import argparse

import os
import pickle
import pandas as pd

from torch import nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from seqeval.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
tqdm.pandas()

from utils import *
from collections import Counter
from torch.utils.data import TensorDataset,\
							 DataLoader,\
							 RandomSampler,\
							 SequentialSampler


from models import *
from alltags import *

####################################################################################################
# Config
####################################################################################################
tag2idx = {t:i for i, t in enumerate(tag_values)}
print(tag_values)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

def construct_attention(tokens, verbcond = False):
	attention = []
	first_idx = None
	for idx, t in enumerate(tokens):
		if not first_idx and t == 0:
			first_idx = idx
		attention.append(float(t != 0.0))

	if verbcond and first_idx > 2:
		attention[first_idx - 2] = 0.0
		attention[first_idx - 1] = 0.0

	return attention

def get_optimizer(model, fulltune = False, lr=5e-5):
	if fulltune:
		param_optimizer = list(model.named_parameters())
		no_decay = ['bias', 'gamma', 'beta']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			 'weight_decay_rate': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
			 'weight_decay_rate': 0.0}
		]
	else:
		param_optimizer = list(model.classifier.named_parameters())
		optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

	return AdamW(
		optimizer_grouped_parameters,
		lr=lr,
		eps=1e-8
	)

####################################################################################################
#
####################################################################################################
def get_tokenizer(FLAGS):
	return BertTokenizer.from_pretrained(FLAGS.tokenizer_bert_name)

def get_dataloader(file, tokenizer, FLAGS):
	sents, labels_ = read_sentences(file, sep = FLAGS.sep)
	tokens, labels, indicators = zip(*[tokenize_and_preserve_labels(s, l, tokenizer, tag2idx, add_special_tokens=True, condition_verb=FLAGS.verbcond) for s,l in zip(sents, labels_)])
	tokens, indicators, labels = pad_and_trim((tokens, indicators), labels, MAX_LEN = FLAGS.MAX_LEN, PAD_VAL = tag2idx['PAD'])

	attention_mask = torch.tensor([[float(i != 0.0) for i in ii] for ii in tokens])
	tokens, labels, indicators = map(torch.tensor, [tokens, labels, indicators])

	data = TensorDataset(tokens, attention_mask, labels, indicators)
	samples = RandomSampler(data)
	dataloader = DataLoader(data, sampler = samples, batch_size=FLAGS.batch_size)

	counts = Counter([y for x in labels_ for y in x])
	class_weights = [sum(counts.values())/counts[t] if counts[t] > 0 else 0. for t in tag_values]
	class_weights = torch.tensor(class_weights).float()

	return dataloader, class_weights

def get_model(FLAGS):
	config = BertConfig.from_pretrained(FLAGS.bert_name)
	config.num_labels = len(tag2idx)
	config.hidden_size_lstm = FLAGS.hidden_size
	config.bidirectional_lstm = FLAGS.bidirectional
	return SimpleBERTLSTM.from_pretrained(FLAGS.bert_name, config=config)

def classification_report(true_labels, predictions):
	true_tags, pred_tags = [], []
	for t, p in zip(true_labels, predictions):
		for t_i, p_i in zip(t, p):
			if tag_values[t_i] not in special_tokens:
				true_tags.append(tag_values[t_i])
				pred_tags.append(tag_values[p_i])

	f1 = f1_score(true_tags, pred_tags)
	acc = accuracy_score(true_tags, pred_tags)

	heads_true, heads_pred = [], []
	for t_i, p_i in zip(true_tags, pred_tags):
		if t_i in heads:
			heads_true.append(t_i)
			if p_i in heads:
				heads_pred.append(p_i)
			else:
				heads_pred.append("O")

	f1_heads = f1_score(heads_true, heads_pred)
	acc_heads = accuracy_score(heads_true, heads_pred)
	confmtx = confusion_matrix(heads_true, heads_pred)

	return f1, acc, f1_heads, acc_heads, confmtx

def train_eval_loop(modelname, train_data, eval_data, model, optimizer, scheduler, FLAGS, **kwargs):
	early_stop, best_f1 = 0, 0

	class_weights = None
	if 'class_weights' in kwargs:
		class_weights = kwargs['class_weights']
		class_weights = class_weights.to(device)

	for epoch in tqdm(range(FLAGS.epochs)):

		###################################################################################33
		# Train
		###################################################################################33
		model.train() # Put model in training mode
		total_loss = 0 # Reset total loss for epoch

		for step, batch in enumerate(train_data):
			batch = tuple(t.to(device) for t in batch) #add batch to gpu

			b_input_ids, b_input_mask, b_labels, b_predind = batch

			model.zero_grad() #Clear previous calculated gradients

			# Forward pass
			outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, pred_indicators=None, class_weights=class_weights)

			loss = outputs[0]
			loss.backward() #backward pass
			total_loss += loss.item()

			#Clip the norm of the gradient to prevent exploding gradients
			torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=FLAGS.max_grad_norm)

			optimizer.step() #update parameters
			scheduler.step() #update learning rate

		# Calculate average loss over train data
		total_loss = total_loss / len(train_data)

		print()
		print("{} Epoch {} train loss: {:.3f}".format(modelname, epoch, total_loss), end=" ")

		###################################################################################33
		# Eval
		###################################################################################33
		model.eval()
		eval_loss = 0
		predictions, true_labels = [], []
		for step, batch in enumerate(eval_data):
			batch = tuple(t.to(device) for t in batch)
			b_input_ids, b_input_mask, b_labels, b_predind = batch

			with torch.no_grad():
				outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, pred_indicators=None, class_weights=class_weights)

			loss = outputs[0]
			eval_loss += loss.item()

			logits = outputs[1].detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			predictions.extend([list(p) for p in np.argmax(logits, axis = 2)])
			true_labels.extend([list(p) for p in label_ids])

		print("val loss: {:.3f}".format(eval_loss), end=" ")

		f1, acc, f1_heads, acc_heads, confmtx = classification_report(true_labels, predictions)
		print("acc: {:.3f} f1: {:.3f} acc-heads: {:.3f} f1-heads: {:.3f}".format(acc, f1, acc_heads, f1_heads))

		if f1_heads > best_f1:

			if FLAGS.debug:
				print()
				print("Val F1-heads improved")
				print(confmtx)
				print()

			early_stop = 0
			best_f1 = f1_heads
			model.save_pretrained(os.path.join(FLAGS.models_dir, "models_{}".format(modelname)))
			np.savetxt(os.path.join(FLAGS.results_dir, "confmatrix_{}".format(modelname),confmtx))

		else:
			early_stop = early_stop + 1
			if early_stop > FLAGS.EARLY_STOP_TH:
				if FLAGS.debug:
					print()
					print("Early stopping")
					print()
					break


####################################################################################################
#
####################################################################################################
def main(argv):
	parser = argparse.ArgumentParser(description='')
	parser.add_argument("modelname", type=str, help = "Name of the model")
	parser.add_argument("train_file", type=str, help = "Train data conll05 format")
	parser.add_argument("test_file", type=str, help = "Test data conll05 format")
	parser.add_argument("--sep", type=str, default=" ||| ", help = "Separator for train/test data")
	parser.add_argument("--tokenizer_bert_name", type=str, default = "bert-base-uncased", help = "Name of the tokenizer to use (see transformers), defaults to bert-base-uncased")
	parser.add_argument("--bert_name", type=str, default = "bert-base-uncased", help = "Name of the BERT model to use (see transformers), defaults to bert-base-uncased")
	parser.add_argument("--MAX_LEN", type=int, default = 105, help = "Maximum sequence length for BERT input (default = 105)")
	parser.add_argument("--batch_size", type=int, default=64, help = "Batch size to train on (default = 64)")
	parser.add_argument("--hidden_size", type=int, default=300, help = "LSTM dimension size")
	parser.add_argument("--epochs", type=int, default=20, help = "Number of epochs to train for")
	parser.add_argument("--EARLY_STOP_TH", type=int, default=5, help = "Threshold for early stopping. Defaults to 5 iterations with no improvement")
	parser.add_argument("--max_grad_norm", type=float, default=1.0, help = "Maximum gradient norm (clipping), defaults to 1.0")
	parser.add_argument("--learning_rate", type=float, default=5e-5, help = "Learning rate (defaults to 5e-5)")
	parser.add_argument("--debug", action='store_true', help = "Verbose output")
	parser.add_argument("--fulltune", action='store_true', help = "Whether to update the BERT weights or not")
	parser.add_argument("--verbcond", action='store_true', help = "Whether to condition BERT models using the identified predicate. Alters BERT's input to [CLS] sentence [SEP] verb [SEP].")
	parser.add_argument("--bidirectional", type=bool, default=True, help = "Whether to use a bidirectional LSTM or not")
	parser.add_argument("--models_dir", type=str, default="models/", help = "Directory where to store trained models")
	parser.add_argument("--results_dir", type=str, default="results/", help = "Directory where to store classification reports")
	parser.add_argument("--no_cuda", action='store_false', help = "Wheter to use CUDA or not")
	FLAGS = parser.parse_args()

	if FLAGS.debug:
		print(FLAGS)

	# Create output directories if not exists
	os.makedirs(FLAGS.models_dir,exist_ok=True)
	os.makedirs(FLAGS.results_dir,exist_ok=True)


	tokenizer = get_tokenizer(FLAGS)

	# Get train data
	train_dataloader, class_weights = get_dataloader(FLAGS.train_file, tokenizer, FLAGS)

	# Get dev data
	dev_dataloader, _ = get_dataloader(FLAGS.dev_file, tokenizer, FLAGS)

	# Get Model
	model = get_model(FLAGS)

	if not FLAGS.no_cuda:
		model.cuda()

	total_steps = len(train_dataloader) * FLAGS.epochs
	optimizer = get_optimizer(model, fulltune=FLAGS.fulltune, lr=FLAGS.learning_rate)

	if FLAGS.debug:
		print(optimizer)

	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=total_steps
	)

	# Training / Eval loop
	train_eval_loop(FLAGS.modelname, train_dataloader, dev_dataloader, model, optimizer, scheduler, FLAGS, class_weights=class_weights)


if __name__ == '__main__':
	main(sys.argv)
