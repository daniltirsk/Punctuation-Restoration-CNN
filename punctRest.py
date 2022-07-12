import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
from torch.nn import functional as F
from transformers import logging

EMBEDDING_DIM = 768
N_FILTERS = 25
FILTER_SIZES = [1, 2, 4, 6, 8, 12, 16]
DROPOUT = 0.5
OUTPUT_DIM = 3

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout, sent_len):
        super().__init__()
        self.sent_len = sent_len

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(sent_len // 2, embedding_dim // fs),
                      stride=embedding_dim // fs // 2
                      )
            for fs in filter_sizes
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(sent_len // 2, embedding_dim // fs),
                      stride=embedding_dim // fs // 2
                      )
            for fs in filter_sizes
        ])

        self.convs3 = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(2, embedding_dim // fs),
                      stride=embedding_dim // fs // 2
                      )
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters * 3, len(filter_sizes) * n_filters)
        self.fc2 = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Softmax(dim=0)

    def forward(self, text):
        embedded = text
        embedded = embedded.unsqueeze(1)

        conved = [F.elu(conv(embedded[:, :, 6 // 2:, :])).squeeze(2) for conv in self.convs]
        conved2 = [F.elu(conv(embedded[:, :, :6 // 2, :])).squeeze(2) for conv in self.convs2]
        conved3 = [F.elu(conv(embedded[:, :, 6 // 2 - 1:6 // 2 + 1, :])).squeeze(2) for conv in self.convs3]

        pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        pooled2 = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved2]
        pooled3 = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved3]

        cat = self.dropout(torch.cat(pooled + pooled2 + pooled3, dim=1))

        logits = self.dropout(self.fc(cat))
        logits = self.fc2(logits)

        return logits


def bert_text_preparation(text, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    # word_lens stores length of each word in input text, because some words get broken into multiple tokens

    word_lens = []
    for i in range(len(tokenized_text)):
        token = tokenized_text[i]
        if token[0:2] == "##":
            word_lens[-1] += 1
        elif tokenized_text[i - 1] == '-':
            word_lens[-2] += 2
            word_lens.pop()
        else:
            word_lens.append(1)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors, word_lens


def get_bert_embeddings(tokens_tensor, segments_tensors, word_lens, model):
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    # Get embeddings from the last 3 layers of BERT and concatenate them
    # For each words that is longer than one token, take the mean of all the token
    # The resulting embedding size is 2304 for each word

    list_token_embeddings = []
    cur_index = 1
    for i in range(1, len(word_lens) - 1):
        new_token = token_embeddings[cur_index:cur_index + word_lens[i], -4:, :]
        new_token = torch.sum(new_token, dim=1)
        new_token = torch.mean(new_token, 0)

        new_token = new_token.flatten()
        list_token_embeddings.append(new_token)
        cur_index += word_lens[i]

    return list_token_embeddings


def prepare_data(train_inputs, train_labels, sent_len=20, embedding_size=2304, device='cpu'):
    train_labels = torch.tensor(train_labels)
    train_labels = train_labels.to(device)

    train_inputs = torch.concat((torch.zeros((sent_len // 2 - 1, embedding_size), ).to(device),
                                 torch.stack(train_inputs).to(device),
                                 torch.zeros((sent_len // 2, embedding_size)).to(device)))

    train_labels = torch.concat((torch.zeros((sent_len // 2 - 1), dtype=torch.long).to(device),
                                 train_labels,
                                 torch.zeros((sent_len // 2), dtype=torch.long).to(device)))

    train_data = TensorDataset(train_inputs, train_labels)

    return train_data


def predict(model, val_data, sent_len, batch_size, device):
    model.eval()

    all_labels = []
    all_preds = []
    for i in range(0, len(val_data) - sent_len, batch_size):
        batch = [tuple(t.to(device) for t in val_data[i + b:i + sent_len + b]) for b in range(batch_size) if
                 i + sent_len + b <= len(val_data)]

        b_input_ids = []
        b_labels = []

        for b in batch:
            input, labels = b
            if len(labels) != sent_len:
                print(labels)
            b_input_ids.append(input)
            all_labels.append(int(labels[sent_len // 2 - 1].item()))
            b_labels.append(labels[sent_len // 2 - 1])

        b_labels = torch.stack(b_labels, dim=0)
        b_input_ids = torch.stack(b_input_ids, dim=0)

        with torch.no_grad():
            logits = model(b_input_ids)

        preds = torch.argmax(logits, dim=1).flatten()

        all_preds += [int(i.item()) for i in preds]

    return all_preds


# convert datasets to X and y
# X contains individual words. Numbers and punctuation marks are left out
# y contains classes for each word, a class is the punctuation mark that comes after a word. For example - "I like NLP!" is [0,0,2]

def process_text(text):
    X = []
    y = []

    tokens = word_tokenize(text)

    seps = [',', '.', '!', '?']

    for token in tokens:

        if token in seps:
            sInd = seps.index(token) + 1
            if sInd >= 2:
                y[-1] = 2
            else:
                y[-1] = 1
        else:
            X.append(token.lower())
            y.append(0)

    return X, y


def load_models(model_path, bert_path, device='cpu'):
    modelPunct = CNN(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, 6)
    modelPunct.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    modelBert = BertModel.from_pretrained(bert_path, local_files_only=True, output_hidden_states=True, )
    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    return modelPunct, modelBert, tokenizer


def restoreText(words, punct):
    seps = [',', '<END>']
    restored = words.copy()
    restored[0] = restored[0].title()
    p_index = np.argwhere(np.array(punct) != 0).flatten()
    for i in p_index:
        restored[i] = restored[i] + seps[punct[i] - 1]
        if i < len(words) - 1 and punct[i] - 1 != 0:
            restored[i + 1] = restored[i + 1].title()

    return ' '.join(restored)

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",
                        "-m",
                        type=str,
                        default='./models/punctRestorationModel.pth')
    parser.add_argument("--bert-path",
                        "-b",
                        type=str,
                        default='./models/bert')
    parser.add_argument("--text", "-t", type=str,
                        default="Тестовый текс. Создан, чтобы проверить работу нейронной сети")
    parser.add_argument("--device", "-d", type=str,
                        default="cpu")

    args = parser.parse_args()

    text = args.text
    model_path = args.model_path

    logging.set_verbosity_error()
    modelPunct, modelBert, tokenizer = load_models(model_path,args.bert_path)

    X, y = process_text(text)

    use_cuda = torch.cuda.is_available() and args.device == "cuda"
    if not use_cuda and args.device == "cuda":
        print("cuda not availabe")

    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = modelPunct.cuda()
    else:
        model = modelPunct.cpu()

    tokenized_text, tokens_tensor, segments_tensors, word_lens = bert_text_preparation(" ".join(X), tokenizer)
    embeddings = get_bert_embeddings(tokens_tensor.to(device), segments_tensors.to(device), word_lens, modelBert)
    data = prepare_data(embeddings, y, 6, EMBEDDING_DIM)
    preds = predict(modelPunct, data, 6, len(y), device)
    print(restoreText(X, preds))