import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

def read_sentences(dataf, sep=" ||| "):
	sentences, labels = [], []
	with open(dataf) as inpt:
		for line in inpt:
			sentence, tags = line.strip().split(sep)

			sentences.append(sentence.split())
			labels.append(tags.split())

		return (sentences, labels)

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer, tag2idx,add_special_tokens=True, condition_verb = False):
	tokenized_sentence = [] if not add_special_tokens else [tokenizer.convert_tokens_to_ids("[CLS]")]
	labels = [] if not add_special_tokens else [tag2idx["CLS"]]
	bv = []

	pred_indicator = [] if not add_special_tokens else [0]

	for word, label in zip(sentence, text_labels):

		# Tokenize the word and count # of subwords the word is broken into
		tokenized_word = tokenizer.tokenize(word)
		tokenized_word = tokenizer.convert_tokens_to_ids(tokenized_word)

		# Add the tokenized word to the final tokenized word list
		tokenized_sentence.extend(tokenized_word)

		# Add the same label to the new list of labels `n_subwords` times
		n_subwords = len(tokenized_word)
		# labels.extend([tag2idx.get(label, tag2idx["O"])] * n_subwords)

		# Fill in subwords with X label
		labels.append(tag2idx[label])
		labels.extend([tag2idx['X']] * (n_subwords - 1))

		if label == "B-V":
			bv = [tokenized_word.copy()]
			pred_indicator.extend([1]*(n_subwords))

		elif label == "I-V":
			bv.append(tokenized_word.copy())
			pred_indicator.extend([1]*(n_subwords))
		else:
			pred_indicator.extend([0]*(n_subwords))

	if add_special_tokens:
		tokenized_sentence.append(tokenizer.convert_tokens_to_ids("[SEP]"))
		labels.append(tag2idx["SEP"])
		pred_indicator.append(0)

	if condition_verb and bv:

		for m, tokenized_word in enumerate(bv):
			tokenized_sentence.extend(tokenized_word)
			n_subwords = len(tokenized_word)
			if m == 0:
				labels.append(tag2idx['B-V'])
				labels.extend([tag2idx['X']] * (n_subwords - 1))
			else:
				labels.append(tag2idx['I-V'])
				labels.extend([tag2idx['X']] * (n_subwords - 1))
			pred_indicator.extend([0]*n_subwords)

		tokenized_sentence.append(tokenizer.convert_tokens_to_ids("[SEP]"))
		labels.append(tag2idx["SEP"])
		pred_indicator.append(0)

	return tokenized_sentence, labels, pred_indicator



# TODO: TRIM DELETES [SEP]!!
def pad_and_trim(zero_pads, *args, MAX_LEN=50, PAD_VAL=0, SEP_VAL = 0):
	res = []
	for l in zero_pads:
		res.append( pad_sequences(l, maxlen=MAX_LEN, dtype="long", value=0.0, truncating="post", padding="post") )
	for l in args:
		res.append( pad_sequences(l, maxlen=MAX_LEN, dtype="long", value=PAD_VAL, truncating="post", padding="post") )
	return res
