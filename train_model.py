import sys
import torch.nn as nn
import torch.optim as optim
from preprocessing import *
from decomp_atten_model import ProjectionNet, DecompAttenModel

EPOCHS = 250
LR = 0.05
WEIGHT_DECAY = 5e-5
EVALUATION_FREQ = 1000

PROJ_MODEL_PATH = './proj_model'
ATTEN_MODEL_PATH = './atten_model'


def train_model(embeds_projection, atten_model, projection_optimizer, atten_optimizer,
                train_batches, dev_batches):
    train_accuracy = 0
    count_train_samples = 0
    correct_train_samples = 0
    accuracy_history = []

    train_sents = 0

    cross_ent_loss = nn.NLLLoss(reduction='mean')

    for i in range(len(train_batches)):
        if i != 0 and i % EVALUATION_FREQ == 0:
            accuracy = loss_accuracy(embeds_projection, atten_model, dev_batches)
            accuracy_history.append(accuracy)
            print('Dev Accuracy: ({:.2f}%)'.format(100. * accuracy))

        embeds_projection.train()
        atten_model.train()

        train_src_batch, train_tgt_batch, train_lbl_batch = train_batches[i]
        train_lbl_batch = train_lbl_batch.type(torch.LongTensor)

        batch_size = train_src_batch.size(0)
        train_sents += batch_size

        projection_optimizer.zero_grad()
        atten_optimizer.zero_grad()

        train_projected_sent1 = embeds_projection(train_src_batch)
        train_projected_sent2 = embeds_projection(train_tgt_batch)

        log_distribution = atten_model(train_projected_sent1, train_projected_sent2)
        loss = cross_ent_loss(log_distribution, train_lbl_batch)
        loss.backward()
        projection_optimizer.step()
        atten_optimizer.step()

        # training accuracy calc
        _, predict = log_distribution.data.max(dim=1)
        count_train_samples += train_lbl_batch.data.size()[0]
        correct_train_samples += torch.sum(predict == train_lbl_batch.data)

    train_accuracy = correct_train_samples / count_train_samples
    print('Train Accuracy: ({:.2f}%)'.format(100. * train_accuracy))

    return train_accuracy, accuracy_history


def loss_accuracy(embeds_projection, atten_model, dev_batches):
    embeds_projection.eval()
    atten_model.eval()
    correct = count = 0.0

    for i in range(len(dev_batches)):
        dev_src_batch, dev_tgt_batch, dev_lbl_batch = dev_batches[i]

        dev_src_linear = embeds_projection(dev_src_batch)
        dev_tgt_linear = embeds_projection(dev_tgt_batch)

        log_distribution = atten_model(dev_src_linear, dev_tgt_linear)

        _, predict = log_distribution.data.max(dim=1)
        count += dev_lbl_batch.data.size()[0]
        correct += torch.sum(predict == dev_lbl_batch.data)

    acc = correct / count

    return acc


def train_and_save(proj_model_file, atten_model_file, glove_file):
    # preprocess
    W2I, train_sent_num, dev_sent_num, test_sent_num = create_words_dict(glove_file)
    L2I = create_labels_dict()

    # training set
    train_batches = create_batches(TRAIN_SENT1, TRAIN_SENT2, TRAIN_LABEL, train_sent_num, W2I, L2I)
    # dev set
    dev_batches = create_batches(DEV_SENT1, DEV_SENT2, DEV_LABEL, dev_sent_num, W2I, L2I)
    # test set
    test_batches = create_batches(TEST_SENT1, TEST_SENT2, TEST_LABEL, test_sent_num, W2I, L2I)

    word_vecs = create_embedding_matrix(W2I, glove_file)
    train_lbl_size = len(list(L2I.keys()))

    print('Preprocessing Done!')

    # build the model
    embeds_projection = ProjectionNet(word_vecs, word_vecs.size(0))
    embeds_projection.embed.weight.requires_grad = False
    atten_model = DecompAttenModel(train_lbl_size)

    # init optimizers
    para1 = filter(lambda p: p.requires_grad, embeds_projection.parameters())
    para2 = atten_model.parameters()

    projection_optimizer = optim.Adagrad(para1, lr=LR, weight_decay=WEIGHT_DECAY)
    atten_optimizer = optim.Adagrad(para2, lr=LR, weight_decay=WEIGHT_DECAY)

    train_accuracy_history = []
    accuracy_history = []

    # main training loop
    for epoch in range(0, EPOCHS):
        print('Epoch {}'.format(epoch))
        train_accuracy, accuracy = train_model(embeds_projection, atten_model, projection_optimizer, atten_optimizer,
                                               train_batches, dev_batches)
        train_accuracy_history.append(train_accuracy)
        accuracy_history += accuracy
    print('Training Done!')

    # test phase
    test_accuracy = loss_accuracy(embeds_projection, atten_model, test_batches)
    print('Test Accuracy: ({:.2f}%)'.format(100. * test_accuracy))

    print('Testing Done!')

    # save trained model
    torch.save(embeds_projection, proj_model_file)
    torch.save(atten_model, atten_model_file)

    return train_accuracy_history, accuracy_history


def main(argv):
    if len(argv) != 2:
        print('Usage: python train_model.py <glove_file_path>')
        return

    glove_file = argv[1]
    train_accuracy_history, accuracy_hist = train_and_save(PROJ_MODEL_PATH, ATTEN_MODEL_PATH, glove_file)

    # save train accuracy history
    with open('train_accuracy_history.txt', 'w') as f:
        print(train_accuracy_history, file=f)

    # save dev accuracy history
    with open('dev_accuracy_history.txt', 'w') as f:
        print(accuracy_hist, file=f)


if __name__ == '__main__':
    main(sys.argv)
