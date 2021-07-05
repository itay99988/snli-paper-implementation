import sys
import numpy as np
import torch

TRAIN_SENT1 = './prep_data/src-train.txt'
TRAIN_SENT2 = './prep_data/targ-train.txt'
TRAIN_LABEL = './prep_data/label-train.txt'

DEV_SENT1 = './prep_data/src-dev.txt'
DEV_SENT2 = './prep_data/targ-dev.txt'
DEV_LABEL = './prep_data/label-dev.txt'

TEST_SENT1 = './prep_data/src-test.txt'
TEST_SENT2 = './prep_data/targ-test.txt'
TEST_LABEL = './prep_data/label-test.txt'

MAX_SEQ_LEN = 100
NUM_OOV_EMBEDS = 100
BATCH_SIZE = 32
GLOVE_DIM = 300

NULL_SYMBOL = "<NULL>"
SEQ_START = "<s>"

# ==========================================Dictionaries==========================================


# our classification problem has 3 classes
def create_labels_dict():
    L2I = {"entailment": 1, "contradiction": 2, "neutral": 3}
    return L2I


# collect words from sent1, sent2
def collect_words_to_dict(sent1_path, sent2_path, W2I, glove_words):
    sent_num = 0

    for _, (src_orig, targ_orig) in enumerate(zip(open(sent1_path, 'r'), open(sent2_path, 'r'))):
        src_orig = src_orig.strip()
        targ_orig = targ_orig.strip()
        targ = targ_orig.strip().split()
        src = src_orig.strip().split()

        if len(targ) > MAX_SEQ_LEN or len(src) > MAX_SEQ_LEN or len(targ) < 1 or len(src) < 1:
            continue

        for word in targ:
            if word not in W2I and word in glove_words:
                W2I[word] = len(W2I) + 1

        for word in src:
            if word not in W2I and word in glove_words:
                W2I[word] = len(W2I) + 1

        sent_num += 1

    return sent_num


def get_glove_words(glove_file):
    glove_words = set()
    for line in open(glove_file, "r", encoding="utf8"):
        word = line.split()[0].strip()
        glove_words.add(word)
    return glove_words


# every word and symbol from training,dev,test set should appear in the dictionary
def create_words_dict(glove_file):
    glove_words = get_glove_words(glove_file)
    W2I = {"<NULL>": 1, "<s>": 2, "<unk>": 3}

    for i in range(1, NUM_OOV_EMBEDS + 1):
        oov_word = '<oov' + str(i) + '>'
        W2I[oov_word] = len(W2I) + 1

    train_num = collect_words_to_dict(TRAIN_SENT1, TRAIN_SENT2, W2I, glove_words)
    print("{} train sentences processed for vocabulary.".format(train_num))
    dev_num = collect_words_to_dict(DEV_SENT1, DEV_SENT2, W2I, glove_words)
    print("{} dev sentences processed for vocabulary.".format(dev_num))
    test_num = collect_words_to_dict(TEST_SENT1, TEST_SENT2, W2I, glove_words)
    print("{} test sentences processed for vocabulary.".format(test_num))

    return W2I, train_num, dev_num, test_num


# ==========================================Sorted indices vectors creation==========================================

def pad(ls, length, symbol):
    if len(ls) >= length:
        return ls[:length]

    return ls + [symbol] * (length -len(ls))


def convert(w, W2I):
    return W2I[w] if w in W2I else W2I['<oov' + str(np.random.randint(1,100)) + '>']


def convert_sequence(ls, W2I):
    return [convert(l, W2I) for l in ls]


def create_batches(srcfile, targetfile, labelfile, num_sents, W2I, L2I):
    max_sent_l = 0
    newseqlength = MAX_SEQ_LEN + 1
    targets = np.zeros((num_sents, newseqlength), dtype=int)
    sources = np.zeros((num_sents, newseqlength), dtype=int)
    labels = np.zeros((num_sents,), dtype=int)
    source_lengths = np.zeros((num_sents,), dtype=int)
    target_lengths = np.zeros((num_sents,), dtype=int)
    both_lengths = np.zeros(num_sents, dtype={'names': ['x', 'y'], 'formats': ['i4', 'i4']})
    dropped = 0
    sent_id = 0
    for _, (src_orig, targ_orig, label_orig) in enumerate(zip(open(srcfile, 'r'), open(targetfile, 'r'), open(labelfile, 'r'))):
        src_orig = src_orig.strip()
        targ_orig = targ_orig.strip()
        targ = [SEQ_START] + targ_orig.split()
        src = [SEQ_START] + src_orig.split()
        label = label_orig.strip().split()
        max_sent_l = max(len(targ), len(src), max_sent_l)
        if len(targ) > newseqlength or len(src) > newseqlength or len(targ) < 2 or len(src) < 2:
            dropped += 1
            continue
        targ = pad(targ, newseqlength, NULL_SYMBOL)
        targ = convert_sequence(targ, W2I)
        targ = np.array(targ, dtype=int)

        src = pad(src, newseqlength, NULL_SYMBOL)
        src = convert_sequence(src, W2I)
        src = np.array(src, dtype=int)

        targets[sent_id] = np.array(targ, dtype=int)
        target_lengths[sent_id] = (targets[sent_id] != 1).sum()
        sources[sent_id] = np.array(src, dtype=int)
        source_lengths[sent_id] = (sources[sent_id] != 1).sum()
        labels[sent_id] = L2I[label[0]]
        both_lengths[sent_id] = (source_lengths[sent_id], target_lengths[sent_id])
        sent_id += 1

    # shuffle all sentences
    rand_idx = np.random.permutation(sent_id)
    targets = targets[rand_idx]
    sources = sources[rand_idx]
    source_lengths = source_lengths[rand_idx]
    target_lengths = target_lengths[rand_idx]
    labels = labels[rand_idx]
    both_lengths = both_lengths[rand_idx]

    # break up batches based on source/target lengths
    source_lengths = source_lengths[:sent_id]

    both_lengths = both_lengths[:sent_id]
    sorted_lengths = np.argsort(both_lengths, order=('x', 'y'))
    sources = sources[sorted_lengths]
    targets = targets[sorted_lengths]
    labels = labels[sorted_lengths]
    target_l = target_lengths[sorted_lengths]
    source_l = source_lengths[sorted_lengths]

    curr_l_src = 0
    curr_l_targ = 0
    l_location = []  # idx where sent length changes

    for j, i in enumerate(sorted_lengths):
        if source_lengths[i] > curr_l_src or target_lengths[i] > curr_l_targ:
            curr_l_src = source_lengths[i]
            curr_l_targ = target_lengths[i]
            l_location.append(j + 1)
    l_location.append(len(sources))

    # get batch sizes
    curr_idx = 1
    batch_idx = [1]
    batch_l = []
    target_l_new = []
    source_l_new = []
    for i in range(len(l_location) - 1):
        while curr_idx < l_location[i + 1]:
            curr_idx = min(curr_idx + BATCH_SIZE, l_location[i + 1])
            batch_idx.append(curr_idx)
    for i in range(len(batch_idx) - 1):
        batch_l.append(batch_idx[i + 1] - batch_idx[i])
        source_l_new.append(source_l[batch_idx[i] - 1])
        target_l_new.append(target_l[batch_idx[i] - 1])

    # create the batches list
    batches = []
    batches_num = len(batch_l)

    for i in range(batches_num):
        sources_batch = torch.from_numpy(np.array(sources[batch_idx[i] : batch_idx[i] + batch_l[i]][:, :source_l_new[i]])) - 1
        targets_batch = torch.from_numpy(np.array(targets[batch_idx[i] : batch_idx[i] + batch_l[i]][:, :target_l_new[i]])) - 1
        labels_batch = torch.from_numpy(np.array(labels[batch_idx[i] : batch_idx[i] + batch_l[i]])) - 1
        batches.append((sources_batch, targets_batch, labels_batch))

    return batches

# ==========================================Embeddings matrix creation==========================================


# load only word that appeared on training
def selectively_load_glove_vec(vocab, glove_file):
    word_vecs = {}
    for line in open(glove_file, 'r', encoding="utf8"):
        d = line.split()
        word = d[0]
        vec = np.array(list(map(float, d[1:])))

        if word in vocab:
            word_vecs[word] = vec
    return word_vecs


def create_embedding_matrix(W2I, glove_file):
    w2v_vecs = np.random.normal(size=(len(W2I), GLOVE_DIM))
    w2v = selectively_load_glove_vec(W2I, glove_file)

    print("num words in pretrained model is " + str(len(w2v)))
    for word, vec in w2v.items():
        w2v_vecs[W2I[word] - 1] = vec
    for i in range(len(w2v_vecs)):
        w2v_vecs[i] = w2v_vecs[i] / np.linalg.norm(w2v_vecs[i])

    return torch.from_numpy(np.array(w2v_vecs))
