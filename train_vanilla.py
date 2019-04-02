import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField
# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=())

train, val, test = torchtext.datasets.SNLI.splits(
    TEXT, LABEL)

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=16, device=torch.device("cuda"), repeat=False)

# Build the vocabulary with word embeddings
# Out-of-vocabulary (OOV) words are hashed to one of 100 random embeddings each
# initialized to mean 0 and standarad deviation 1 (Sec 5.1)
import random
unk_vectors = [torch.randn(300) for _ in range(100)]
TEXT.vocab.load_vectors(vectors='glove.6B.300d',
                        unk_init=lambda x:random.choice(unk_vectors))
# normalized to have l_2 norm of 1
vectors = TEXT.vocab.vectors
vectors = vectors / vectors.norm(dim=1,keepdim=True)
vectors = NamedTensor(vectors, ('word', 'embedding'))
TEXT.vocab.vectors = vectors

def test_code(model):
    "All models should be able to be run with following command."
    model.eval()
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10, device=torch.device("cuda"))
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.premise, batch.hypothesis)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
        upload += argmax.tolist()

    with open("predictions.txt", "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")

import os.path

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0")
path = "data/"

embedding_dim = 300
nclasses = len(LABEL.vocab)

def train_model(model, train_iter, optimizer, criterion, every=1000, save=False, epoch=0, best_val_loss=1E9):
    model.train()
    key = model.key
    total_loss=0.
    num_batches=0.
    total=0.
    total_right=0.
    for b, batch in enumerate(train_iter):
        optimizer.zero_grad()
        output = model(batch.premise, batch.hypothesis)
        preds = output.max('output')[1]
        preds_eq = preds.eq(batch.label)
        loss = 0.
        loss += criterion(output, batch.label).values
        loss.backward()
        optimizer.step()
        num_batches += 1
        total += float(batch.premise.shape['batch'])
        total_right += preds_eq.sum().item()
        total_loss += loss.detach().item()*float(batch.premise.shape['batch'])
        torch.cuda.empty_cache() 
        if(b%every == 0):
            print('[B{:4d}] Train Loss: {:.3e}'.format(b, total_loss/total))
            if save:
                torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
                }, path+'models'+key+'E{:4d}B{:4d}'.format(epoch, b))
    return total_right / total, total_loss / total

def test_model(model, test_iter, criterion, num_batches=-1):
    model.eval()
    with torch.no_grad():
        total = 0
        total_right = 0
        total_loss = 0
        for b, batch in enumerate(test_iter):
            if(not(num_batches==-1) and b > num_batches):
                break
            output = model(batch.premise, batch.hypothesis)
            preds = output.max('output')[1]
            preds_eq = preds.eq(batch.label)
            loss = 0.
            loss += criterion(output, batch.label).values
            total += float(batch.premise.shape['batch'])
            total_right += preds_eq.sum().item()
            total_loss += loss.detach().item() * float(batch.premise.shape['batch'])
    return total_right / total, total_loss / total

def load_model(model, version=''):
    # model with lowest validation loss thus far is saved at path+'models'+key
    # we can also load a specific version, i.e. path+'models'+key+'E20B2000'
    # not for training! Doesn't load an optimizer.
    key = model.key
    # load checkpoint
    fname = path+'models'+key+version
    if os.path.isfile(fname):
        checkpoint = torch.load(fname)
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        best_val_loss = checkpoint['best_val_loss']
        print("Loaded Checkpoint:", epoch_start, best_val_loss, np.exp(best_val_loss))

def checkpoint_trainer(model, optimizer, criterion, version='', nepochs=20):
    # model with lowest validation loss thus far is saved at path+'models'+key
    # we can also load a specific version, i.e. path+'models'+key+'E20B2000'
    key = model.key
    best_val_loss = -1
    epoch_start = 0

    # load checkpoint
    fname = path+'models'+key+version
    if os.path.isfile(fname):
        checkpoint = torch.load(fname)
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['best_val_loss']
        print("Loaded Checkpoint:", epoch_start, best_val_loss, np.exp(best_val_loss))

    for epoch in range(epoch_start, nepochs):
        train_acc, train_loss = train_model(model, train_iter, 
                                            optimizer,
                                            criterion,
                                            every=1000,
                                            epoch=epoch,
                                            best_val_loss=best_val_loss)
        val_acc, val_loss = test_model(model, val_iter, criterion)
        print('[E{:4d}] | Train Acc: {:.3e} Train Loss: {:.3e} | Val Acc: {:.3e} Val Loss: {:.3e} PPL: {:.3e}'.format(epoch, train_acc, train_loss, val_acc, val_loss, np.exp(val_loss)))
        if(val_loss < best_val_loss or best_val_loss == -1):
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                }, path+'models'+key)

class FeedForwardReLU(ntorch.nn.Module):
    def __init__(self, input_name, input_dim, output_name, output_dim=200, hidden_dim=200, dropout=0.2):
        super(FeedForwardReLU, self).__init__()
        self.input_name = input_name
        self.output_name = output_name
        self.lineara = ntorch.nn.Linear(input_dim, hidden_dim).spec(input_name, 'hidden')
        self.dropouta = ntorch.nn.Dropout(dropout)
        self.linearb = ntorch.nn.Linear(hidden_dim, output_dim).spec('hidden', output_name)
        self.dropoutb = ntorch.nn.Dropout(dropout)
    def forward(self, x):
        x = self.lineara(x).relu()
        x = self.dropouta(x)
        x = self.linearb(x).relu()
        x = self.dropoutb(x)
        return x

class DecomposableAttentionCore(ntorch.nn.Module):
    def __init__(self, representation_dim=embedding_dim, hidden_dim=200):
        super(DecomposableAttentionCore, self).__init__()
        self.F = FeedForwardReLU('representation', representation_dim, 'attention', hidden_dim)
        self.G = FeedForwardReLU('representation', representation_dim * 2, 'comparison', hidden_dim)
        self.H1 = FeedForwardReLU('comparison', hidden_dim * 2, 'hidden', hidden_dim)
        self.H2 = ntorch.nn.Linear(hidden_dim, nclasses).spec('hidden', 'output')
        # attention visualization
        self.attnWeightsAlpha = None
        self.attnWeightsBeta = None
    def forward(self, abar, bbar):
        # abar: batch, premiseseqlen, representation
        # bbar: batch, premiseseqlen, representation
        adecomposed = self.F(abar) # batch, premiseseqlen, attention
        bdecomposed = self.F(bbar) # batch, hypothesisseqlen, attention
        e = adecomposed.dot('attention', bdecomposed) # batch, premiseseqlen, hypothesisseqlen
        self.attnWeightsAlpha = e.softmax('premiseseqlen') # batch, premiseseqlen, hypothesisseqlen
        self.attnWeightsBeta = e.softmax('hypothesisseqlen') # batch, premiseseqlen, hypothesisseqlen
        alpha = self.attnWeightsAlpha.dot('premiseseqlen', abar) # batch, hypothesisseqlen, representation
        beta = self.attnWeightsBeta.dot('hypothesisseqlen', bbar) # batch, premisesseqlen, representation
        aBeta = ntorch.cat((abar, beta), 'representation') # batch, premisesseqlen, representation * 2
        bAlpha = ntorch.cat((bbar, alpha), 'representation') # batch, hypothesisseqlen, representation * 2
        v1dot = self.G(aBeta) # batch, premiseseqlen, comparison
        v2dot = self.G(bAlpha) # batch, hypothesisseqlen, comparison
        v1 = v1dot.sum('premiseseqlen') # batch, comparison
        v2 = v2dot.sum('hypothesisseqlen') # batch, comparison
        v = ntorch.cat((v1, v2), 'comparison') # batch, comparison * 2
        yhat = self.H2(self.H1(v)) # batch, output
        return yhat

class DecomposableAttentionVanilla(ntorch.nn.Module):
    def __init__(self, key, embedding_dim=embedding_dim, hidden_dim=200):
        super(DecomposableAttentionVanilla, self).__init__()
        self.key = key
        self.embedding = ntorch.nn.Embedding(len(TEXT.vocab), embedding_dim).spec('seqlen', 'representation')
        self.embedding.weight.data.copy_(TEXT.vocab.vectors.values)
        self.core = DecomposableAttentionCore(embedding_dim, hidden_dim)
        # attention visualization
        self.attnWeightsAlpha = None
        self.attnWeightsBeta = None
    def forward(self, a, b):
        # a: batch, seqlen
        # b: batch, seqlen
        abar = self.embedding(a).rename('seqlen', 'premiseseqlen') # batch, premiseseqlen, embedding
        bbar = self.embedding(b).rename('seqlen', 'hypothesisseqlen') # batch, hypothesisseqlen, embedding
        yhat = self.core(abar, bbar) # batch, output
        self.attnWeightsAlpha = self.core.attnWeightsAlpha
        self.attnWeightsBeta = self.core.attnWeightsBeta
        return yhat

model = DecomposableAttentionVanilla('4.1.vanilla.v1').to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05, lr_decay=0, weight_decay=0, initial_accumulator_value=0.1)
criterion = ntorch.nn.CrossEntropyLoss().spec("output")

# model with lowest validation loss thus far is saved at path+'models'+key
# we can also load a specific version, i.e. path+'models'+key+'E20B2000'
version = ''

checkpoint_trainer(model, optimizer, criterion, version)
