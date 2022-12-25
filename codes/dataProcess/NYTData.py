# coding: utf-8

import io
import os
import sys
import warnings

import numpy as np
from gluonnlp.data import BERTSentenceTransform
from mxnet.gluon.data import SimpleDataset
from tqdm import tqdm

current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_directory)
sys.path.append(root_path)

from config import Config

warnings.filterwarnings('ignore')

__all__ = ['BagDataset', 'BagDatasetTransform', 'BERTDatasetTransform',
           'load_rel', 'bag_batchify_fn', 'bagDataset_batchify_fn']

config = Config()


def load_rel(rel2id_path: str) -> (dict, list):
    """加载样本标签

    Args:
        rel2id_path: str
            path of relation2id file
    Returns:
        rel2id: dict
        id2rel: list
    """

    rel2id = {}
    id2rel = []
    with open(rel2id_path) as fr:
        fr.readline()
        for rel in fr:
            rel, index = rel.split(' ')
            rel2id[rel] = int(index)
            id2rel.append(rel)
    return rel2id, id2rel


def load_entity(ent2id_path: str) -> (dict, list):
    """

    Args:
        ent2id_path: str
            path of entity2id file
    Returns:
        ent2id: dict
        id2ent: list
    """

    ent2id = {}
    id2ent = []
    with open(ent2id_path) as fr:
        num = int(fr.readline().strip())
        for i in range(num):
            ent, ent_id = fr.readline().strip().split('\t')
            ent2id[ent] = int(ent_id)
            id2ent.append(ent)
    return ent2id, id2ent


class BagDataset(SimpleDataset):
    """Dataset that composed by bags.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    is_train: bool, default True
        indicate the data is training data or test data.
    encoding : str, default 'utf8'
        File encoding format.

    Returns
    -------
    bag: list
        the value is:
        [(triple info): left entity, right entity, label, ids of sentences;
        [(sentence info): left entity, right entity, left entity pos, right entity pos, label, sentence length;
        input token ids;
        left entity relative distant, '51' denotes -1, '50' denotes 0, '49' denotes 1, ...;
        right entity relative distant;
        piecewise mask]; [...]; ...]
    """

    def __init__(self, filename, is_train: bool = True, encoding='utf8'):
        bags = []
        self.is_train = is_train
        with io.open(filename, 'r', encoding=encoding) as in_file:
            for line in in_file:
                bag = self.__get_bag__(line.strip(), in_file)
                bags.append(bag)
        super(BagDataset, self).__init__(bags)

    def __get_bag__(self, bag_info, in_file):
        bag = []
        bag_info = bag_info.split('\t')
        bag.append(bag_info)
        if self.is_train:
            sentence_num = len(bag_info[3].split(','))
        else:
            sentence_num = len(bag_info[2].split(','))
        for i in range(sentence_num):
            sentence = []
            for _ in range(5):
                sentence.append(in_file.readline().strip())
            bag.append(sentence)

        return bag


class IntraBag(object):
    def __init__(self, entity_pair):
        self.entity_pair = entity_pair
        self.sen_ids = []
        self.sen_info = []


class IntraBagDataset(SimpleDataset):
    """Dataset that composed by bags.
        this data version followed this paper:
        [NAACL 2019] Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions

    Parameters
    ----------
    filename : str
        Path to the input text file.
    is_train: bool, default True
        indicate the data is training data or test data.
    encoding : str, default 'utf8'
        File encoding format.

    Returns
    -------
    bag: list[IntraBag]

    """

    def __init__(self, filename, is_train: bool = True, encoding='utf8'):
        bags = dict()
        total_sen_num = 0
        self.is_train = is_train
        with io.open(filename, 'r', encoding=encoding) as in_file:
            for line in in_file:
                bag_info = line.strip().split('\t')
                ent_pair = (bag_info[0], bag_info[1])
                if ent_pair not in bags.keys():
                    bags[ent_pair] = IntraBag(ent_pair)
                current_Bag = bags.get(ent_pair)
                sen_ids, bag, sen_num = self.__get_bag__(bag_info, in_file)
                current_Bag.sen_ids += sen_ids
                current_Bag.sen_info += bag
                total_sen_num += sen_num
        print(total_sen_num)
        super(IntraBagDataset, self).__init__(list(bags.values()))

    def __get_bag__(self, bag_info: list, in_file):
        bag = []
        if self.is_train:
            sen_ids = bag_info[3].split(',')
        else:
            sen_ids = bag_info[2].split(',')
        sentence_num = len(sen_ids)
        for i in range(sentence_num):
            sentence = []
            for _ in range(5):
                sentence.append(in_file.readline().strip())
            bag.append(sentence)
        return sen_ids, bag, sentence_num


def cut_tokens(fixLen_sen: int, tokens_list: list, leftPos: int, rightPos: int) -> (list, int, int):
    """cut tokens if sentence length is longer than max length, expand words from two entities

    Args:
        fixLen_sen: int
            fix length of sentence
        tokens_list: list
            tokens, e.g. words, mask, wpe, tokens...
            In pcnn+att/cnn+att, it is words, left wpe, right wpe and mask, eg. [words, left_wpe, right_wpe, mask]
            In bert, is [tokens]
        leftPos: int
            position of left entity
        rightPos: int
            position of right entity
    Returns:
        new_tokens_list: list[list]
            tokens include two entities
        new_leftPos: int
        new_rightPos: int

    """
    # get first tokens
    first_tokens = tokens_list[0]
    new_tokens_list = []
    new_leftPos = leftPos
    new_rightPos = rightPos
    # longer than FixLen, expand from two entities
    if len(first_tokens) > fixLen_sen:
        # sentence between two entities longer than FixLen
        if rightPos - leftPos + 1 > fixLen_sen:
            for tokens in tokens_list:
                new_tokens_list.append(tokens[leftPos:leftPos + fixLen_sen])
            new_leftPos = 0
            new_rightPos = fixLen_sen - 1
        # sentence between two entities shorter than FixLen
        else:
            # words between two entities
            for tokens in tokens_list:
                new_tokens_list.append(tokens[leftPos:rightPos + 1])
            # boundaries on both sides
            before = leftPos
            after = rightPos + 1
            new_leftPos = 0
            new_rightPos = rightPos - leftPos

            left_len = fixLen_sen - len(new_tokens_list[0])
            # words before left entity
            backward = int((fixLen_sen - len(new_tokens_list[0])) / 2)
            # words after left entity
            forward = left_len - backward

            if leftPos - forward < 0:
                before_start = 0
                before_end = before
                after_start = after
                after_end = after + left_len - before
            elif rightPos + backward >= len(first_tokens):
                before_start = before - (left_len - len(first_tokens[after:]))
                before_end = before
                after_start = after
                after_end = len(first_tokens)
            else:
                before_start = before - forward
                before_end = before
                after_start = after
                after_end = after + backward
            new_leftPos = new_leftPos + (before_end - before_start)
            new_rightPos = new_rightPos + (before_end - before_start)
            # expand words from two entities
            for i, tokens in enumerate(new_tokens_list):
                new_tokens_list[i] = tokens_list[i][before_start:before_end] + \
                                     tokens + \
                                     tokens_list[i][after_start:after_end]

    else:
        for tokens in tokens_list:
            new_tokens_list.append(tokens)
    return new_tokens_list, new_leftPos, new_rightPos


class BagDatasetTransform(object):
    """Dataset Transform for bag level relation extraction

    Parameters
    ----------
    rel2id: dict
        relation to index.
    ent2id: dict
        entity to index.
    fixLen_sen: int
        the fix length of sentences.
    is_train: bool, default True
        indicate the data is training data or test data.
    use_cls: bool, default False
        whether to use BERT style CLS token

    Returns
    -------
    ids: np.ndarray, dtype=int32, shape is (sen_num, )
        the list of sentence index.
    sen_num: np.ndarray, dtype=int32, shape is (1, )
        the sentence number in current bag.
    bag_entId: np.ndarray, dtype=int32, shape is (1, 2)
        the entities' KG id in the bag.
    bag_entPos: np.ndarray, dtype=int32, shape is (sen_num, 2)
        the entities' position index in every sentence.
    bag_inputs: np.ndarray, dtype=int32, shape is (sen_num, fixLen_sen)
        the indexes of tokens in sentence.
    bag_leftDist: np.ndarray, dtype=int32, shape is (sen_num, fixLen_sen)
        the relative distance from left entity.
    bag_rightDist: np.ndarray, dtype=int32, shape is (sen_num, fixLen_sen)
        the relative distance from right entity.
    bag_mask: np.ndarray, dtype=int32, shape is (sen_num, fixLen_sen)
        the mask index used in pcnn.
    bag_valid_length: np.ndarray, dtype=int32, shape is (sen_num, )
        the valid_length of every sentence in bag.
    labels: np.ndarray, dtype=int32, shape is (1, ) in training or (4, ) in testing.
        the labels of bag. train data has only one label, and the test data has at most 4 labels,
        0 denotes 'NA', -1 denotes None
    """

    def __init__(self, rel2id: dict, ent2id: dict, fixLen_sen: int, is_train: bool = True, use_cls: bool = False):
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.is_train = is_train
        self.use_cls = use_cls
        self.cls = config.vocab_cls_size - 1
        if self.use_cls:
            self.fixLen_sen = fixLen_sen - 1
        else:
            self.fixLen_sen = fixLen_sen

    def __get_pad_feature__(self, words, leftPos, rightPos, leftDist, rightDist, mask):

        # get tokens between two entities

        new_tokens_list, new_leftPos, new_rightPos = cut_tokens(self.fixLen_sen,
                                                                [words, leftDist, rightDist, mask],
                                                                leftPos,
                                                                rightPos)
        wordsWithPad, leftDistWithPad, rightDistWithPad, maskWithPad = new_tokens_list
        pad = [0] * 80
        dist_pad = [101] * 80

        wordsWithPad = wordsWithPad + pad[:self.fixLen_sen - len(wordsWithPad)]
        maskWithPad = maskWithPad + pad[:self.fixLen_sen - len(maskWithPad)]
        leftDistWithPad = leftDistWithPad + dist_pad[:self.fixLen_sen - len(leftDistWithPad)]
        rightDistWithPad = rightDistWithPad + dist_pad[:self.fixLen_sen - len(rightDistWithPad)]

        if self.use_cls:
            wordsWithPad = [self.cls] + wordsWithPad
            maskWithPad = [0] + maskWithPad
            leftDistWithPad = [101] + leftDistWithPad
            rightDistWithPad = [101] + rightDistWithPad

        new_tokens_list = [np.array(wordsWithPad, dtype='int32'), np.array(leftDistWithPad, dtype='int32'),
                           np.array(rightDistWithPad, dtype='int32'), np.array(maskWithPad, dtype='int32')]
        return new_tokens_list, new_leftPos, new_rightPos

    def __call__(self, bag: list):
        bag_inputs = []
        bag_entId = []
        bag_entWordID = []
        bag_entPos = []
        bag_leftDist = []
        bag_rightDist = []
        bag_mask = []
        bag_valid_length = []
        labels = set()
        headEntity = bag[0][0]
        tailEntity = bag[0][1]
        # add entity id
        bag_entId.append([self.ent2id[headEntity], self.ent2id[tailEntity]])
        if self.is_train:
            labels.add(self.rel2id.get(bag[0][2], 0))
            ids = list(map(int, bag[0][3].split(',')))
        else:
            ids = list(map(int, bag[0][2].split(',')))
        for i, _ in enumerate(ids, start=1):
            # 取句子的标签和长度
            _, _, head_pos, tail_pos, label, sen_len = bag[i][0].split(',')
            sen_len = int(sen_len)
            left_pos = int(head_pos)
            right_pos = int(tail_pos)
            is_reverse = False
            # get valid_length
            if self.use_cls:
                sen_len = sen_len + 1

            bag_valid_length.append(sen_len if sen_len <= self.fixLen_sen else self.fixLen_sen)
            # 如果是测试集，取每个句子的标签
            if not self.is_train:
                labels.add(int(label))
            words = list(map(int, bag[i][1].split(',')))
            left_dist = list(map(int, bag[i][2].split(',')))
            right_dist = list(map(int, bag[i][3].split(',')))
            mask = list(map(int, bag[i][4].split(',')))
            # 由于 triple_info 中，所给的实体信息是按照三元组的顺序给出的，
            # 所以当 triple_info 中头实体位置靠后时，只互换二者位置，用于后续的句子切分
            # 但是dist不需要互换，WPE起到了用来指示关系中实体方向的作用
            bag_entWordID.append([words[left_pos], words[right_pos]])
            if left_pos > right_pos:
                is_reverse = True
                # pos互换
                temp_pos = left_pos
                left_pos = right_pos
                right_pos = temp_pos

            new_tokens_list, new_leftPos, new_rightPos = self.__get_pad_feature__(words, left_pos, right_pos,
                                                                                  left_dist, right_dist, mask)
            wordsWithPad, leftDistWithPad, rightDistWithPad, maskWithPad = new_tokens_list
            # add entity position
            # 判断实体位置之前是否进行过反转，用于区分实体方向
            if is_reverse:
                bag_entPos.append([new_rightPos, new_leftPos])
            else:
                bag_entPos.append([new_leftPos, new_rightPos])
            bag_inputs.append(wordsWithPad)
            bag_leftDist.append(leftDistWithPad)
            bag_rightDist.append(rightDistWithPad)
            bag_mask.append(maskWithPad)

        labels = list(labels)
        # 对测试集补齐标签，一个bag默认最多4个标签
        if not self.is_train:
            assert len(labels) <= 4, "The labels of bag are more than 4, (%s,%s)" % (headEntity, tailEntity)
            pad = [-1] * 4
            labels = labels + pad[len(labels):]
        return np.array(ids, dtype='int32'), np.array([len(ids)], dtype='int32'), \
               np.array(bag_entId, dtype='int32'), np.array(bag_entWordID, dtype='int32'), \
               np.array(bag_entPos, dtype='int32'), np.array(bag_inputs, dtype='int32'), \
               np.array(bag_leftDist, dtype='int32'), np.array(bag_rightDist, dtype='int32'), \
               np.array(bag_mask, dtype='int32'), np.array(bag_valid_length, dtype='int32'), \
               np.array(labels, dtype='int32')


class IntraBagDatasetTransform(object):
    """IntraBagDataset Transform for bag level relation extraction
        This version is for intra-bag which is from
        [NAACL 2019] Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions

    Parameters
    ----------
    rel2id: dict
        relation to index.
    ent2id: dict
        entity to index.
    fixLen_sen: int
        the fix length of sentences.
    is_train: bool, default True
        indicate the data is training data or test data.

    Returns
    -------
    ids: np.ndarray, dtype=int32, shape is (sen_num, )
        the list of sentence index.
    sen_num: np.ndarray, dtype=int32, shape is (1, )
        the sentence number in current bag.
    bag_entID: np.ndarray, dtype=int32, shape is (1, 2)
        the entities' KG id in the bag.
    bag_entWordID：np.ndarray, dtype=int32, shape is (sen_num, 2)
        the entities' word id in the bag.
    bag_entPos: np.ndarray, dtype=int32, shape is (sen_num, 2)
        the entities' position index in every sentence.
    bag_inputs: np.ndarray, dtype=int32, shape is (sen_num, fixLen_sen)
        the indexes of tokens in sentence.
    bag_leftDist: np.ndarray, dtype=int32, shape is (sen_num, fixLen_sen)
        the relative distance from left entity.
    bag_rightDist: np.ndarray, dtype=int32, shape is (sen_num, fixLen_sen)
        the relative distance from right entity.
    bag_mask: np.ndarray, dtype=int32, shape is (sen_num, fixLen_sen)
        the mask index used in pcnn.
    bag_valid_length: np.ndarray, dtype=int32, shape is (sen_num, )
        the valid_length of every sentence in bag.
    labels: np.ndarray, dtype=int32, shape is (1, ) in training or (4, ) in testing.
        the labels of bag. train data has only one label, and the test data has at most 4 labels,
        0 denotes 'NA', -1 denotes None
    """

    def __init__(self, rel2id: dict, ent2id: dict, fixLen_sen: int, is_train: bool = True):
        self.rel2id = rel2id
        self.ent2id = ent2id
        self.is_train = is_train
        self.fixLen_sen = fixLen_sen

    def __get_pad_feature__(self, words, leftPos, rightPos, leftDist, rightDist, mask):

        # get tokens between two entities
        new_tokens_list, new_leftPos, new_rightPos = cut_tokens(self.fixLen_sen,
                                                                [words, leftDist, rightDist, mask],
                                                                leftPos,
                                                                rightPos)
        wordsWithPad, leftDistWithPad, rightDistWithPad, maskWithPad = new_tokens_list
        pad = [0] * 80
        dist_pad = [101] * 80

        wordsWithPad = wordsWithPad + pad[:self.fixLen_sen - len(wordsWithPad)]
        maskWithPad = maskWithPad + pad[:self.fixLen_sen - len(maskWithPad)]
        leftDistWithPad = leftDistWithPad + dist_pad[:self.fixLen_sen - len(leftDistWithPad)]
        rightDistWithPad = rightDistWithPad + dist_pad[:self.fixLen_sen - len(rightDistWithPad)]

        new_tokens_list = [np.array(wordsWithPad, dtype='int32'), np.array(leftDistWithPad, dtype='int32'),
                           np.array(rightDistWithPad, dtype='int32'), np.array(maskWithPad, dtype='int32')]
        return new_tokens_list, new_leftPos, new_rightPos

    def __call__(self, bag: IntraBag):
        bag_inputs = []
        bag_entID = []
        bag_entPos = []
        bag_leftDist = []
        bag_rightDist = []
        bag_mask = []
        bag_valid_length = []
        labels = set()
        headEntity = bag.entity_pair[0]
        tailEntity = bag.entity_pair[1]
        # add entity id
        bag_entID.append([self.ent2id[headEntity], self.ent2id[tailEntity]])
        # labels.add(self.rel2id.get(bag[0][2], 0))
        ids = bag.sen_ids
        for i in range(len(ids)):
            current_bag = bag.sen_info[i]
            # 取句子的标签和长度
            _, _, head_pos, tail_pos, label, sen_len = current_bag[0].split(',')
            sen_len = int(sen_len)
            left_pos = int(head_pos)
            right_pos = int(tail_pos)
            is_reverse = False
            # get valid_length
            bag_valid_length.append(sen_len if sen_len <= self.fixLen_sen else self.fixLen_sen)
            labels.add(int(label))
            words = list(map(int, current_bag[1].split(',')))
            left_dist = list(map(int, current_bag[2].split(',')))
            right_dist = list(map(int, current_bag[3].split(',')))
            mask = list(map(int, current_bag[4].split(',')))
            # 由于 triple_info 中，所给的实体信息是按照三元组的顺序给出的，
            # 所以当 triple_info 中头实体位置靠后时，只互换二者位置，用于后续的句子切分
            # 但是dist不需要互换，WPE起到了用来指示关系中实体方向的作用
            if left_pos > right_pos:
                is_reverse = True
                # pos互换
                temp_pos = left_pos
                left_pos = right_pos
                right_pos = temp_pos
            new_tokens_list, new_leftPos, new_rightPos = self.__get_pad_feature__(words, left_pos, right_pos,
                                                                                  left_dist, right_dist, mask)
            wordsWithPad, leftDistWithPad, rightDistWithPad, maskWithPad = new_tokens_list
            # add entity position
            # 判断实体位置之前是否进行过反转，用于区分实体方向
            if is_reverse:
                bag_entPos.append([new_rightPos, new_leftPos])
            else:
                bag_entPos.append([new_leftPos, new_rightPos])
            bag_inputs.append(wordsWithPad)
            bag_leftDist.append(leftDistWithPad)
            bag_rightDist.append(rightDistWithPad)
            bag_mask.append(maskWithPad)

        labels = list(labels)
        # 对测试集补齐标签，一个bag默认最多4个标签
        if not self.is_train:
            assert len(labels) <= 4, "The labels of bag are more than 4, (%s,%s)" % (headEntity, tailEntity)
            pad = [-1] * 4
            labels = labels + pad[len(labels):]
        bag_woLabel = [np.array(ids, dtype='int32'), np.array([len(ids)], dtype='int32'),
                       np.array(bag_entID, dtype='int32'), np.array(bag_entPos, dtype='int32'),
                       np.array(bag_inputs, dtype='int32'), np.array(bag_leftDist, dtype='int32'),
                       np.array(bag_rightDist, dtype='int32'), np.array(bag_mask, dtype='int32'),
                       np.array(bag_valid_length, dtype='int32')]
        bag_wLabel = []
        if self.is_train:
            for label in labels:
                bag_wLabel.append(bag_woLabel + [np.array([label], dtype='int32')])
        else:
            bag_wLabel.append(bag_woLabel + [np.array(labels, dtype='int32')])
        return bag_wLabel


class BERTDatasetTransform(object):
    """Dataset Transformation for BERT-style Sentence Classification or Regression.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    labels : list of str, int, float or None. defaults None
        List of all labels for the classification task and regressing task.
        If labels is None, the default task is regression
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    label_dtype: int32 or float32, default int32
        label_dtype = int32 for classification task
        label_dtype = float32 for regression task
    is_train: bool, default True
        the data is training set or testing set
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 labels=None,
                 pad=True,
                 pair=True,
                 label_dtype='int32',
                 is_train=True):

        self.label_dtype = label_dtype
        self.labels = labels
        self.is_train = is_train
        self.fixLen_sen = max_seq_length - 20
        if self.labels:
            self._label_map = {}
            for (i, label) in enumerate(labels):
                self._label_map[label] = i
        self._bert_xform = BERTSentenceTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)

    def __call__(self, bag):
        """Perform transformation for sequence pairs or single sequences.

        The transformation of one sentence in bag is processed in the following steps:
        - cut sentences if sentences is longer than fixed length
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        ## BERTSentenceTransform ##
        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.
        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0

        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1

        Parameters
        ----------
        bag: list
            [e1id, e2id, label, sentences_ids]
            [[e1name, e2name, left_pos, right_pos, label, legth],
            [text],
            [left_dist],
            [right_dist],
            [mask]]
            ...
        Returns
        -------
        np.array: sentences ids in 'int32' ,shape (batch_size, sen_num)
        np.array: input token ids in 'int32', shape (batch_size, sen_num, seq_length)
        np.array: valid length in 'int32', shape (batch_size, sen_num)
        np.array: input token type ids in 'int32', shape (batch_size, sen_num, seq_length)
        np.array: classification task: label id in 'int32',
                  if tarin set, shape (batch_size, 1, 1),
                  if test set, shape (batch_size, 4, 1)
        """

        bag_inputs = []
        bag_length = []
        bag_segment = []
        labels = set()
        leftEntity = bag[0][0]
        rightEntity = bag[0][1]

        if self.is_train:
            labels.add(self._label_map.get(bag[0][2], 0))
            ids = list(map(int, bag[0][3].split(',')))
        else:
            ids = list(map(int, bag[0][2].split(',')))

        for i, _ in enumerate(ids, start=1):
            # 取句子的标签和长度
            _, _, left_pos, right_pos, label, _ = bag[i][0].split(',')
            left_pos = int(left_pos)
            right_pos = int(right_pos)
            # 如果是测试集，取每个句子的标签
            if not self.is_train:
                labels.add(int(label))
            # cut words need to convert sentence to word list
            words = bag[i][1].split()

            # 当triple_info 中左边实体位置靠后时，只互换二者位置，用于后续的句子切分
            # 但是dist不需要互换，WPE起到了用来指示关系中实体方向的作用
            if left_pos > right_pos:
                # pos互换
                temp_pos = left_pos
                left_pos = right_pos
                right_pos = temp_pos

            # 'cut_tokens' return a list, so need to get the first element in list
            words = cut_tokens(self.fixLen_sen, [words], left_pos, right_pos)[0]
            # make words list convert to sentence
            words = ' '.join(words)
            input_ids, valid_length, segment_ids = self._bert_xform(tuple([words]))
            bag_inputs.append(input_ids)
            bag_length.append(valid_length)
            bag_segment.append(segment_ids)

        labels = list(labels)
        # 对测试集补齐标签，一个bag默认最多4个标签
        if not self.is_train:
            assert len(labels) <= 4, "The labels of bag are more than 4, (%s,%s)" % (leftEntity, rightEntity)
            pad = [-1] * 4
            labels = labels + pad[len(labels):]

        # convert to numpy type
        ids = np.array(ids, dtype='int32')
        bag_inputs = np.array(bag_inputs, dtype='int32')
        bag_length = np.array(bag_length, dtype='int32')
        bag_segment = np.array(bag_segment, dtype='int32')
        labels = np.array(labels, dtype=self.label_dtype)

        return ids, bag_inputs, bag_segment, bag_length, labels


def generate_OneTwoTest(data: list):
    """generate one sentence bag and two sentences bag from Bag(more than one sentence)

    Args:
        data: list
            All test data

    Returns:

    """
    oneSen_bags = []
    twoSen_bags = []
    for bag_info in data:
        if bag_info[1][0] <= 1:
            continue
        one_index = np.random.choice(bag_info[1][0], 1, replace=False)
        two_index = np.random.choice(bag_info[1][0], 2, replace=False)
        oneSen_info = []
        twoSen_info = []
        # ids
        oneSen_info.append(bag_info[0][one_index])
        twoSen_info.append(bag_info[0][two_index])
        # sen_num
        oneSen_info.append(np.array([1]))
        twoSen_info.append(np.array([2]))
        # bag_entID
        oneSen_info.append(bag_info[2])
        twoSen_info.append(bag_info[2])
        for info in bag_info[3:-1]:
            oneSen_info.append(info[one_index])
            twoSen_info.append(info[two_index])
        # label
        oneSen_info.append(bag_info[-1])
        twoSen_info.append(bag_info[-1])
        oneSen_bags.append(oneSen_info)
        twoSen_bags.append(twoSen_info)

    return oneSen_bags, twoSen_bags


def genrate_AllTrueTest(data: list):
    AllTrueTest = []
    for bag_info in data:
        if bag_info[-1][0] == 0 or bag_info[1][0] <= 1:
            continue
        AllTrueTest.append(bag_info)

    return AllTrueTest


def bagDataset_batchify_fn(data):
    """旧版本的batchify_fn, 针对lazy_DataSet,
    由于每个batch size都需要预处理, 影响效率，故弃用修改为一次性预处理
    lazy_DataSet 每一行为tuple, 需要先通过*data解包，再利用zip将对应列进行拼接->zip(*data)
    zip拼接结果为tuple，所以再通过[*data]转化为list
    Args:
        data: SimpleDataset

    Returns:
        list：[[id: np.array], [bag: np.array], [label: np.array]]
    """

    if isinstance(data[0], np.ndarray):
        return [*data]
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [bagDataset_batchify_fn(i) for i in data]
    else:
        return np.asarray(data)


def bag_batchify_fn(data) -> list:
    """新版batchify_fn, 针对一次性预处理后的数据

    Args:
        data: np.NDArray
        一次性预处理后的数据, 格式为np.NDArray,
        [ids, sen_num, bag_entId, bag_entPos, bag_inputs, bag_leftDist, bag_rightDist, bag_mask, bag_valid_length, labels]
        shape is (batch_size, 10)

    Returns:
        list: [[ids: np.array], [sen_num: np.array], [bag_inputs: np.array],......,[label: np.array]]

    """

    if isinstance(data[0], np.ndarray):
        # zip(*data) 对齐列，[[*i] for i in data] 再将tuple转为list
        dataOrganizedByCol = zip(*data)
        return [list(i) for i in dataOrganizedByCol]
    else:
        raise Exception('wrong data type')


def generate_BagData():
    """
    [ids, sen_num, bag_entId, bag_entPos bag_inputs, bag_leftDist, bag_rightDist, bag_mask, bag_valid_length, labels]
    """

    # load data
    print('——————load data——————\n')
    rel2id, id2rel = load_rel(config.rel2id_path)
    ent2id_520K, id2ent_520K = load_entity(config.ent2id_path_520K)
    ent2id_570K, id2ent_570K = load_entity(config.ent2id_path_570K)

    trainDataSet_570K = BagDataset(config.bag_train_path_570K, is_train=True)
    trainTransform_570K = BagDatasetTransform(rel2id, ent2id_570K, config.fixLen_sen, is_train=True, use_cls=False)
    trainDataSet_570K = trainDataSet_570K.transform(trainTransform_570K)

    testDataSet_570K = BagDataset(config.bag_test_path, is_train=False)
    testTransform_570K = BagDatasetTransform(rel2id, ent2id_570K, config.fixLen_sen, is_train=False, use_cls=False)
    testDataSet_570K = testDataSet_570K.transform(testTransform_570K)

    trainDataSet_520K = BagDataset(config.bag_train_path_520K, is_train=True)
    trainTransform_520K = BagDatasetTransform(rel2id, ent2id_520K, config.fixLen_sen, is_train=True, use_cls=False)
    trainDataSet_520K = trainDataSet_520K.transform(trainTransform_520K)

    testDataSet_520K = BagDataset(config.bag_test_path, is_train=False)
    testTransform_520K = BagDatasetTransform(rel2id, ent2id_520K, config.fixLen_sen, is_train=False, use_cls=False)
    testDataSet_520K = testDataSet_520K.transform(testTransform_520K)


    train_570K = list(map(lambda x: x, tqdm(trainDataSet_570K)))
    test_all_570K = list(map(lambda x: x, tqdm(testDataSet_570K)))
    train_520K = list(map(lambda x: x, tqdm(trainDataSet_520K)))
    test_all_520K = list(map(lambda x: x, tqdm(testDataSet_520K)))

    train_570K = np.array(train_570K)
    test_all_570K = np.array(test_all_570K)
    train_520K = np.array(train_520K)
    test_all_520K = np.array(test_all_520K)

    print('——————save train 570K data——————')
    print('shape:', train_570K.shape)
    np.save(os.path.join(root_path, 'data/NYT_data/NYT_570088/train_570K.npy'), train_570K)

    print('——————save train 520K data——————')
    print('shape:', train_520K.shape)
    np.save(os.path.join(root_path, 'data/NYT_data/NYT_522611/train_520K.npy'), train_520K)

    print('——————save test data——————')
    print('shape', test_all_570K.shape)
    np.save(os.path.join(root_path, 'data/NYT_data/NYT_570088/test_570K_All.npy'), test_all_570K)

    print('shape', test_all_520K.shape)
    np.save(os.path.join(root_path, 'data/NYT_data/NYT_522611/test_520K_All.npy'), test_all_520K)

    print('done')

if __name__ == '__main__':
    # 生成普通数据集
    generate_BagData()