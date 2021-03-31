import torch
from torch import nn

from transformers import BertTokenizer, BertModel, BertForTokenClassification,BertPreTrainedModel

############################################################################
#
############################################################################
class SimpleBERTLSTM(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.lstm = nn.GRU(config.hidden_size,
						    config.hidden_size_lstm,
							num_layers = 1,
							bidirectional = config.bidirectional_lstm)
		self.classifier = nn.Linear(2 * config.hidden_size_lstm if config.bidirectional_lstm else config.hidden_size_lstm, config.num_labels)
		self.num_labels = config.num_labels
		self.init_weights()

	def forward(self,
				input_ids=None,
				attention_mask=None,
				labels=None,
				class_weights=None,
				pred_indicators=None):

		outputs = self.bert(
			input_ids,
			token_type_ids=pred_indicators,
			attention_mask=attention_mask
		)

		sequence_output = self.dropout(outputs[0])
		hidden_states = self.lstm(sequence_output)
		hidden_states = hidden_states[0]

		logits = self.classifier(hidden_states)
		outputs = (logits, ) #TODO: add hidden_states, attention if they are there

		if labels is not None:
			loss_fct = nn.CrossEntropyLoss(weight=class_weights)
			# Only keep active parts of the loss
			if attention_mask is not None:
				active_loss = attention_mask.view(-1) == 1
				active_logits = logits.view(-1, self.num_labels)
				active_labels = torch.where(
					active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
				)
				loss = loss_fct(active_logits, active_labels)
			else:
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

			outputs = (loss, ) + outputs

		return outputs
