import numpy as np
import collections
import torch
from torch import nn
import torch.nn.functional as F

GLOVE_DIM = 300
PROJECTION_DIM = 300
HIDDEN_MLP_DIM = 300
PARAM_STD = 0.01


# ========================================== Projection Network ==========================================
class ProjectionNet(nn.Module):
    def __init__(self, init_embed, words_count):
        super(ProjectionNet, self).__init__()
        self.embed = nn.Embedding(words_count, GLOVE_DIM)
        self.embed.weight.data.copy_(torch.tensor(init_embed))

        self.proj_matrix = nn.Linear(GLOVE_DIM, PROJECTION_DIM, bias=False)
        # parameters init
        torch.nn.init.normal_(self.proj_matrix.weight, mean=0.0, std=PARAM_STD)

    def forward(self, x):
        batch_size = x.size()[0]
        embeds = self.embed(x)

        # embeds structure after conversion: (batches * num_of_words) * (glove_dim)
        embeds = embeds.view(-1, GLOVE_DIM)

        result = self.proj_matrix(embeds)

        # result structure after conversion: batches * num_of_words * glove_dim
        result = result.view(batch_size, -1, PROJECTION_DIM)
        return result

# ========================================== Decomposable Attention Model ==========================================


class DecompAttenModel(nn.Module):
    def __init__(self, label_count):
        super(DecompAttenModel, self).__init__()

        # f function for the "attend" step
        self.f_func = self.single_mlp(PROJECTION_DIM)
        # g function for the "compare" step
        self.g_func = self.single_mlp(PROJECTION_DIM*2)
        # h function for the "aggregate" step
        self.h_func = self.single_mlp(HIDDEN_MLP_DIM*2)
        # final linear layer for the "aggregate" step
        self.final_linear_layer = nn.Linear(HIDDEN_MLP_DIM, label_count, bias=True)

        self.log_prob = nn.LogSoftmax(dim=1)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, PARAM_STD)
                m.bias.data.normal_(0, PARAM_STD)

    def single_mlp(self, input_size):
        relu_stack = nn.Sequential(
                                    nn.Dropout(0.2),
                                    nn.Linear(input_size, HIDDEN_MLP_DIM, bias=True),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(HIDDEN_MLP_DIM, HIDDEN_MLP_DIM, bias=True),
                                    nn.ReLU()
                                  )
        return relu_stack

    def forward(self, sent1_batch, sent2_batch):
        sent1_batch_size, sent1_len = sent1_batch.size(0), sent1_batch.size(1)
        sent2_batch_size, sent2_len = sent2_batch.size(0), sent2_batch.size(1)

        # "attend" step
        f_sent1 = self.f_func(sent1_batch.view(-1, PROJECTION_DIM))
        f_sent1 = f_sent1.view(sent1_batch_size, -1, HIDDEN_MLP_DIM) # batch_size * sent_len * hidden_dim
        f_sent2 = self.f_func(sent2_batch.view(-1, PROJECTION_DIM))
        f_sent2 = f_sent2.view(sent2_batch_size, -1, HIDDEN_MLP_DIM) # batch_size * sent_len * hidden_dim

        f_sent1_T = torch.transpose(f_sent1, 1, 2)
        eij = torch.bmm(f_sent2, f_sent1_T)
        softmax_eij = F.softmax(eij, dim=2)
        sent2_lincom = torch.bmm(torch.transpose(softmax_eij, 1, 2), sent2_batch)
        sent1_with_subphrases = torch.cat((sent1_batch, sent2_lincom), 2)

        eji = torch.transpose(eij, 1, 2)
        softmax_eji = F.softmax(eji, dim=2)
        sent1_lincom = torch.bmm(torch.transpose(softmax_eji, 1, 2), sent1_batch)
        sent2_with_subphrases = torch.cat((sent2_batch, sent1_lincom), 2)

        # "compare" step
        g_sent1 = self.g_func(sent1_with_subphrases.view(-1, PROJECTION_DIM*2))
        g_sent1 = g_sent1.view(sent1_batch_size, -1, HIDDEN_MLP_DIM) # batch_size * sent_len * hidden_dim
        g_sent2 = self.g_func(sent2_with_subphrases.view(-1, PROJECTION_DIM*2))
        g_sent2 = g_sent2.view(sent2_batch_size, -1, HIDDEN_MLP_DIM) # batch_size * sent_len * hidden_dim

        # "aggregate step"
        v1 = torch.sum(g_sent1, dim=1)
        v1 = v1.squeeze(1)
        v2 = torch.sum(g_sent2, dim=1)
        v2 = v2.squeeze(1)

        v1v2 = torch.cat((v1, v2), 1)
        h_v1v2 = self.h_func(v1v2)
        h_v1v2 = h_v1v2.view(sent1_batch_size, HIDDEN_MLP_DIM) # batch_size * hidden_dim

        final_layer_output = self.final_linear_layer(h_v1v2)

        # according to the paper, the cross entropy loss is calculated over the *log* softmax of the score of each class
        log_distribution = self.log_prob(final_layer_output)

        return log_distribution
