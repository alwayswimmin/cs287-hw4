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
        # _, argmax = probs.max('classes')
        _, argmax = probs.max('output')
        upload += argmax.tolist()

    with open("predictions.txt", "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")

import os.path

import matplotlib.pyplot as plt
import numpy as np

from namedtensor.distributions import ndistributions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            total_loss += loss.detach().item()
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

class DecomposableAttentionIntra(ntorch.nn.Module):
    def __init__(self, key, embedding_dim=embedding_dim, hidden_dim=200, biasparamsperside=11):
        super(DecomposableAttentionIntra, self).__init__()
        self.key = key
        self.embedding = ntorch.nn.Embedding(len(TEXT.vocab), embedding_dim).spec('seqlen', 'representation')
        self.embedding.weight.data.copy_(TEXT.vocab.vectors.values)
        self.Fintra = FeedForwardReLU('representation', embedding_dim, 'hidden', hidden_dim)
        self.biasparamsperside = biasparamsperside
        if self.biasparamsperside != -1:
            # 3D vector to dance around ReplicationPad1d only working for 3D, 4D, 5D
            self.biasparams = torch.nn.Parameter(torch.zeros(1, 1, 2 * biasparamsperside + 1), requires_grad=True)
        self.core = DecomposableAttentionCore(embedding_dim * 2, hidden_dim)
        # attention visualization
        self.aselfAttnWeights = None
        self.bselfAttnWeights = None
    def distance_bias_matrix(self, seqlen):
        npadding = max(0, seqlen - self.biasparamsperside - 1)
        start = npadding + self.biasparamsperside
        m = torch.nn.ReplicationPad1d(npadding)
        padded = m(self.biasparams).squeeze()
        row_list = []
        for ii in range(seqlen):
            row_list.append(torch.roll(padded, ii - start))
        extended = torch.stack(row_list)
        return extended[:, :seqlen]
    def forward(self, a, b):
        # a: batch, seqlen
        # b: batch, seqlen
        aembedding = self.embedding(a).rename('seqlen', 'premiseseqlen') # batch, premiseseqlen, embedding
        bembedding = self.embedding(b).rename('seqlen', 'hypothesisseqlen') # batch, hypothesisseqlen, embedding
        adecomposedintra = self.Fintra(aembedding) # batch, premiseseqlen, hidden
        bdecomposedintra = self.Fintra(bembedding) # batch, hypothesisseqlen, hidden
        # the following renaming dance is necessary to distinguish between two otherwise equivalent dimensions in a square matrix.
        adecomposedintra2 = adecomposedintra.rename('premiseseqlen', 'premiseseqlen2') # batch, premiseseqlen2, hidden
        bdecomposedintra2 = bdecomposedintra.rename('hypothesisseqlen', 'hypothesisseqlen2') # batch, hypothesisseqlen2, hidden
        af = adecomposedintra.dot('hidden', adecomposedintra2) # batch, premiseseqlen, premiseseqlen2
        bf = bdecomposedintra.dot('hidden', bdecomposedintra2) # batch, hypothesisseqlen, hypothesisseqlen2
        ad = bd = 0
        if self.biasparamsperside != -1:
            ad = NamedTensor(self.distance_bias_matrix(a.shape['seqlen']), ('premiseseqlen', 'premiseseqlen2')) # premiseseqlen, premiseseqlen2
            bd = NamedTensor(self.distance_bias_matrix(b.shape['seqlen']), ('hypothesisseqlen', 'hypothesisseqlen2')) # hypothesisseqlen, hypothesisseqlen2
        self.aselfAttnWeights = (af + ad).softmax('premiseseqlen') # batch, premiseseqlen, premiseseqlen2
        self.bselfAttnWeights = (bf + bd).softmax('hypothesisseqlen') # batch, hypothesisseqlen, hypothesisseqlen2
        aprime = self.aselfAttnWeights.dot('premiseseqlen', aembedding) # batch, premiseseqlen2, embedding
        bprime = self.bselfAttnWeights.dot('hypothesisseqlen', bembedding) # batch, hypothesisseqlen2, embedding
        aprime = aprime.rename('premiseseqlen2', 'premiseseqlen') # batch, premiseseqlen, embedding
        bprime = bprime.rename('hypothesisseqlen2', 'hypothesisseqlen') # batch, hypothesisseqlen, embedding
        abar = ntorch.cat((aembedding, aprime), 'representation') # batch, premiseseqlen, embedding * 2
        bbar = ntorch.cat((bembedding, bprime), 'representation') # batch, hypothesisseqlen, embedding * 2
        yhat = self.core(abar, bbar) # batch, output
        self.attnWeightsAlpha = self.core.attnWeightsAlpha
        self.attnWeightsBeta = self.core.attnWeightsBeta
        return yhat

def visualize(model, batch):
    model.eval()
    # a: batch, seqlen
    # b: batch, seqlen
    intra = isinstance(model, DecomposableAttentionIntra)
    yhat = model(batch.premise, batch.hypothesis)
    for batchnum in range(batch.premise.shape['batch']):
        premise = batch.premise.get("batch", batchnum)
        hypothesis = batch.hypothesis.get("batch", batchnum)
        label = batch.label.get("batch", batchnum)
        attnWeightsAlpha = model.attnWeightsAlpha.get("batch", batchnum) # premiseseqlen, hypothesisseqlen
        attnWeightsBeta = model.attnWeightsBeta.get("batch", batchnum) # premiseseqlen, hypothesisseqlen
        if intra:
            aselfAttnWeights = model.aselfAttnWeights.get("batch", batchnum) # premiseseqlen, premiseseqlen
            bselfAttnWeights = model.bselfAttnWeights.get("batch", batchnum) # hypothesisseqlen, hypothesisseqlen
        
        premise = [TEXT.vocab.itos[i] for i in premise.tolist()]
        hypothesis = [TEXT.vocab.itos[i] for i in hypothesis.tolist()]
        label = LABEL.vocab.itos[label.item()]
        
        # title, ylabel, ytickslabels, xlabel, xticklabels, data
        graph_info = []
        graph_info.append(('$\\alpha$ Attention', 'premise', premise, 'hypothesis', hypothesis, attnWeightsAlpha.cpu().detach().numpy()))
        graph_info.append(('$\\beta$ Attention', 'premise', premise, 'hypothesis', hypothesis, attnWeightsBeta.cpu().detach().numpy()))
        if intra:
            graph_info.append(('$a$ Self-Attention', 'premise', premise, 'premise', premise, aselfAttnWeights.cpu().detach().numpy()))
            graph_info.append(('$b$ Self-Attention', 'hypothesis', hypothesis, 'hypothesis', hypothesis, bselfAttnWeights.cpu().detach().numpy()))
        
        for title, ylabel, row_labels, xlabel, column_labels, data in graph_info:
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

            # put the major ticks at the middle of each cell
            ax.set_xticks(np.arange(data.shape[1]) + 0.5)
            ax.set_yticks(np.arange(data.shape[0]) + 0.5)

            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            plt.xticks(rotation=60)

            ax.set_title(title)
            ax.set_xticklabels(column_labels)
            ax.set_xlabel(xlabel)
            ax.set_yticklabels(row_labels)
            ax.set_ylabel(ylabel)

            plt.show()

class MixtureModel(ntorch.nn.Module):
    def __init__(self, key, K):
        super(MixtureModel, self).__init__()
        self.key = key
        self.K = K
        self.models = torch.nn.ModuleList([DecomposableAttentionIntra(key+str(k)) for k in range(K)])
        
    def forward(self, a, b):
        yhats = [self.models[k](a,b) for k in range(self.K)]
        avg_yhat = self.models[0](a,b)
        for k in range(1,self.K):
            avg_yhat += self.models[k](a,b)
        avg_yhat /= self.K
        return avg_yhat

class QInference(ntorch.nn.Module):
    def __init__(self, K, representation_dim=embedding_dim, hidden_dim=200):
        super(QInference, self).__init__()
        self.K = K
        self.embedding = ntorch.nn.Embedding(len(TEXT.vocab), embedding_dim).spec('seqlen', 'representation')
        self.embedding.weight.data.copy_(TEXT.vocab.vectors.values)
        self.F = FeedForwardReLU('representation', representation_dim, 'attention', hidden_dim)
        self.G = FeedForwardReLU('representation', representation_dim * 2, 'comparison', hidden_dim)
        self.H1 = FeedForwardReLU('comparison', hidden_dim * 2 + hidden_dim, 'hidden', hidden_dim)
        self.H2 = ntorch.nn.Linear(hidden_dim, K).spec('hidden', 'output')
        # attention visualization
        self.attnWeightsAlpha = None
        self.attnWeightsBeta = None
        self.yembedding = ntorch.nn.Embedding(len(LABEL.vocab), hidden_dim).spec('batch', 'comparison')
    def forward(self, a, b, y):
        # abar: batch, premiseseqlen, representation
        # bbar: batch, premiseseqlen, representation
        abar = self.embedding(a).rename('seqlen', 'premiseseqlen') # batch, premiseseqlen, embedding
        bbar = self.embedding(b).rename('seqlen', 'hypothesisseqlen') # batch, hypothesisseqlen, embedding
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
        y_emb = self.yembedding(y)
        v = ntorch.cat((v1, v2, y_emb), 'comparison') # batch, comparison * 2
        yhat = self.H2(self.H1(v)) # batch, output
        return yhat.softmax('output')

class VAE(ntorch.nn.Module):
    def __init__(self, key, K):
        super(VAE, self).__init__()
        self.key = key
        self.K = K
        self.models = torch.nn.ModuleList([DecomposableAttentionIntra(key+str(k)) for k in range(K)])
        
    def forward(self, a, b, c=None):
        if self.training:
            l = [(c.get('batch', i).item(), a.get('batch', i), b.get('batch', i)) for i in range(c.shape['batch'])]
            y_hat = ntorch.stack([self.models[ci](ai, bi) for ci, ai, bi in l], 'batch')
            return y_hat
        yhats = [self.models[k](a,b) for k in range(self.K)]
        avg_yhat = self.models[0](a,b)
        for k in range(1,self.K):
            avg_yhat += self.models[k](a,b)
        avg_yhat /= self.K
        return avg_yhat


def train_model_vae(q_inference, model, train_iter, optimizer, criterion, every=100, save=False, epoch=0, best_val_loss=1E9):
    q_inference.train()
    model.train()
    key = model.key
    total_loss=0.
    num_batches=0.
    total=0.
    total_right=0.
    for b, batch in enumerate(train_iter):
        optimizer.zero_grad()
        premise, hypothesis, label = batch.premise, batch.hypothesis, batch.label
        probs = q_inference(premise, hypothesis, label)
        var_posterior = ndistributions.Categorical(logits=probs.log(), dim_logit='output')
        c = var_posterior.sample() # no gradients can be backpropagated further here!
        # c_probs = probs.index_select('output', c) # z * probs + (1-z) * (1-probs) # ?
        c_probs = ntorch.stack([probs[{'batch':i, 'output':c[{'batch':i}].item()}] for i in range(c.shape['batch'])],'batch')
        
        output = model(premise, hypothesis, c)
        batch_size = output.size('batch')

        nll_raw = criterion(output, label)
        nll = nll_raw.sum()
        prior = ndistributions.Categorical(
            NamedTensor(torch.ones(batch_size, K) / K, ('batch', 'output')).to(device)
        )
        kl = ndistributions.kl_divergence(var_posterior, prior).sum()

        l = nll.item()
        k = kl.item()
        kl_weight = 1.0 # min(1.0, (float(i)/kl_anneal_steps)**2)
        reinforce_term = (nll_raw.detach() * c_probs.log()).sum()

        loss = nll.values + kl * kl_weight + reinforce_term.values
        loss.backward()
        optimizer.step()
        
        preds = output.max('output')[1]
        preds_eq = preds.eq(batch.label)
        
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
                'q_inference': q_inference.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
                }, path+'models'+key+'E{:4d}B{:4d}'.format(epoch, b))
    return total_right / total, total_loss / total

def test_model_vae(q_inference, model, test_iter, criterion, num_batches=-1):
    q_inference.eval()
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
            loss += criterion(output, batch.label).sum()
            total += float(batch.premise.shape['batch'])
            total_right += preds_eq.sum().item()
            total_loss += loss.detach().item()
    return total_right / total, total_loss / total

def load_model_vae(q_inference, model, version=''):
    # model with lowest validation loss thus far is saved at path+'models'+key
    # we can also load a specific version, i.e. path+'models'+key+'E20B2000'
    # not for training! Doesn't load an optimizer.
    key = model.key
    # load checkpoint
    fname = path+'models'+key+version
    if os.path.isfile(fname):
        checkpoint = torch.load(fname)
        epoch_start = checkpoint['epoch'] + 1
        q_inference.load_state_dict(checkpoint['q_inference'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['best_val_loss']
        print("Loaded Checkpoint:", epoch_start, best_val_loss, np.exp(best_val_loss))

def checkpoint_trainer_vae(q_inference, model, optimizer, criterion, version='', nepochs=20, every=1000):
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
        q_inference.load_state_dict(checkpoint['q_inference'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['best_val_loss']
        print("Loaded Checkpoint:", epoch_start, best_val_loss, np.exp(best_val_loss))

    for epoch in range(epoch_start, nepochs):
        train_acc, train_loss = train_model_vae(q_inference, model, train_iter, 
                                                optimizer,
                                                criterion,
                                                every=every,
                                                epoch=epoch,
                                                best_val_loss=best_val_loss)
        val_acc, val_loss = test_model_vae(q_inference, model, val_iter, criterion)
        print('[E{:4d}] | Train Acc: {:.3e} Train Loss: {:.3e} | Val Acc: {:.3e} Val Loss: {:.3e} PPL: {:.3e}'.format(epoch, train_acc, train_loss, val_acc, val_loss, np.exp(val_loss)))
        if(val_loss < best_val_loss or best_val_loss == -1):
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'q_inference': q_inference.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                }, path+'models'+key)

"""#### Scratch"""

K = 3
q_inference = QInference(K).to(device)
model = VAE('4.5.vae.v1', K).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.025, lr_decay=0, weight_decay=0, initial_accumulator_value=0.1)
criterion = ntorch.nn.CrossEntropyLoss(reduce=False).spec("output")

load_model_vae(q_inference, model)
# acc, loss = test_model_vae(q_inference, model, test_iter, criterion)
# print('Acc: {:.3e} Loss: {:.3e}'.format(acc, loss))
# test_code(model)

def interpret_model_vae(q_inference, model, test_iter, criterion, num_batches=-1):
    q_inference.eval()
    model.eval()
    with torch.no_grad():
        total = 0
        total_right = 0
        total_loss = 0
        total_probs = [NamedTensor(torch.zeros(K), 'output').to(device) for i in range(nclasses)]
        counts = [0 for i in range(nclasses)]
        for b, batch in enumerate(test_iter):
            if(not(num_batches==-1) and b > num_batches):
                break
            premise, hypothesis, label = batch.premise, batch.hypothesis, batch.label
            probs = q_inference(premise, hypothesis, label)
            for p, l in [(probs[{'batch':i}], label[{'batch':i}].item()) for i in range(probs.shape['batch'])]:
                total_probs[l] += p
                counts[l] += 1
        avg_probs = [t / c for t, c in zip(total_probs, counts)]
    return avg_probs

# print(interpret_model_vae(q_inference, model, test_iter, criterion))

def interpret_model_vae_2(q_inference, model, test_iter, criterion, num_batches=-1):
    q_inference.eval()
    model.eval()
    with torch.no_grad():
        total = 0
        total_right = 0
        total_loss = 0
        sz = 60
        total_probs = [NamedTensor(torch.zeros(K), 'output').to(device) for i in range(sz)]
        counts = [0 for i in range(sz)]
        for b, batch in enumerate(test_iter):
            if(not(num_batches==-1) and b > num_batches):
                break
            premise, hypothesis, label = batch.premise, batch.hypothesis, batch.label
            probs = q_inference(premise, hypothesis, label)
            for p, a, b, l in [(probs[{'batch':i}], premise[{'batch':i}], hypothesis[{'batch':i}], label[{'batch':i}].item()) for i in range(probs.shape['batch'])]:
                num_words = (a.ne(TEXT.vocab.stoi['<pad>'])).values.sum().item()
                try:
                    total_probs[num_words] += p
                    counts[num_words] += 1
                except:
                    print(num_words)
        avg_probs = [t / c for t, c in zip(total_probs, counts)]
    return avg_probs

import math
for ii, probs in enumerate(interpret_model_vae_2(q_inference, model, test_iter, criterion)):
    p = probs[{'output':0}].item()
    if not math.isnan(p):
        print(ii, p)

