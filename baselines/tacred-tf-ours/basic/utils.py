# util functions
import data_utils

def _get_feed_dict(model, x_batch, y_batch, x_lens, use_pos=False, use_ner=False, use_signature=False):
    feed = {model.word_inputs:x_batch[data_utils.WORD_FIELD], model.labels:y_batch, model.seq_lens:x_lens}
    if use_pos:
        feed[model.pos_inputs] = x_batch[data_utils.POS_FIELD]
    if use_ner:
        feed[model.ner_inputs] = x_batch[data_utils.NER_FIELD]
    if use_signature:
        b = get_signature_batch(x_batch[data_utils.SUBJ_NER_FIELD], x_batch[data_utils.OBJ_NER_FIELD])
        feed[model.signature_inputs] = b
    return feed

def get_signature_batch(subj_batch, obj_batch):
    ''' Get a batch of signature inputs from two separate batch of subj/obj sequences.'''
    b = []
    for subj_seq, obj_seq in zip(subj_batch, obj_batch):
        s, o = data_utils.UNK_ID, data_utils.UNK_ID
        for x in subj_seq:
            if x is not None:
                s = x
                break
        for x in obj_seq:
            if x is not None:
                o = x
                break
        b.append([s,o])
    return b

def _get_feed_dict_with_position_seq_separate_padding(model, x_batch, y_batch, x_lens, use_pos=False, use_ner=False, use_signature=False):
    # use (-sent_len-1) for padding
    sent_len = len(x_batch[data_utils.WORD_FIELD][0])
    return _get_feed_dict_with_position_seq(model, x_batch, y_batch, x_lens, -sent_len-1, use_pos, use_ner, use_signature)

def _get_feed_dict_with_position_seq_zero_padding(model, x_batch, y_batch, x_lens, use_pos=False, use_ner=False, use_signature=False):
    # use 0 for padding position index; this is used when the model can ignore the padding indices using the seq length, otherwise
    # it may be confused with entity position, which is also 0.
    return _get_feed_dict_with_position_seq(model, x_batch, y_batch, x_lens, 0, use_pos, use_ner, use_signature)

def _get_feed_dict_with_position_seq(model, x_batch, y_batch, x_lens, pad_index, use_pos=False, use_ner=False, use_signature=False):
    '''Feed in not only word sequence, but also the position of each word relative to subj and obj and sequences. 
    This is used primarily for CNN model. '''
    feed = {model.word_inputs:x_batch[data_utils.WORD_FIELD], model.labels:y_batch, model.seq_lens:x_lens}
    # The subj and obj sequences from the loaded files are not converted into ids except for PAD token
    # they are in the form of [ None, None, ..., SUBJ, ... , OBJ, ... None, 0, 0, 0, ...],
    # where 0 is the PAD id
    subj_batch, obj_batch = x_batch[data_utils.SUBJ_FIELD], x_batch[data_utils.OBJ_FIELD]
    sent_len = len(subj_batch[0])
    subj_position_batch, obj_position_batch = [], []
    for subj_seq, obj_seq, seq_len in zip(subj_batch, obj_batch, x_lens):
        # create subj position sequence
        subj_idx = find_all_index_in_list(subj_seq, 'SUBJECT')
        subj_position_seq = generate_position_seq(sent_len, subj_idx[0], subj_idx[-1], seq_len, pad_index) # pad output should be 1 smaller than the smallest number
        # create obj position sequence
        obj_idx = find_all_index_in_list(obj_seq, 'OBJECT')
        obj_position_seq = generate_position_seq(sent_len, obj_idx[0], obj_idx[-1], seq_len, pad_index)

        subj_position_batch.append(subj_position_seq)
        obj_position_batch.append(obj_position_seq)
    # add in position batch
    feed[model.subj_pos_inputs] = subj_position_batch
    feed[model.obj_pos_inputs] = obj_position_batch

    if use_pos:
        feed[model.pos_inputs] = x_batch[data_utils.POS_FIELD]
    if use_ner:
        feed[model.ner_inputs] = x_batch[data_utils.NER_FIELD]
    if use_signature:
        b = get_signature_batch(x_batch[data_utils.SUBJ_NER_FIELD], x_batch[data_utils.OBJ_NER_FIELD])
        feed[model.signature_inputs] = b
    return feed

def find_all_index_in_list(l, target):
    idx = []
    for i,x in enumerate(l):
        if x == target:
            idx.append(i)
    return idx

def generate_position_seq(sent_len, min_ent_idx, max_ent_idx, seq_len, pad_output):
    position_seq = []
    for i in range(sent_len):
        if i >= seq_len:
            position_seq.append(pad_output)
        elif i < min_ent_idx:
            position_seq.append(i-min_ent_idx)
        elif i > max_ent_idx:
            position_seq.append(i-max_ent_idx)
        else:
            position_seq.append(0)
    return position_seq
