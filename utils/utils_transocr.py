import torch

global alp2num_character, alphabet_character
alp2num_character = None

def converter(label):
    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            text_input[i][j + 1] = alp2num[label[i][j]]

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])

    
    return length, text_input, text_all, None, None, None, string_label





def converter_ocr_bbox(label, loc_str_tuple):
    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character
    loc_str = [i for i in loc_str_tuple]

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    loc_gt = torch.zeros(batch, max_length).cuda()
    for i in range(batch):
        loc_tmps = [float(s) for s in loc_str[i].split('_')]
        for j in range(len(label[i]) - 1):
            text_input[i][j + 1] = alp2num[label[i][j]]
            loc_gt[i][j+1] = loc_tmps[j]

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])
        
    return length, text_input, text_all, None, None, None, string_label, loc_gt

def converter_ocr(label):
    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            text_input[i][j + 1] = alp2num[label[i][j]]

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])
        
    return length, text_input, text_all, None, None, None, string_label


def get_alphabet(alpha_path):
    global alp2num_character, alphabet_character
    if alp2num_character == None:
        alphabet_character_file = open(alpha_path)
        alphabet_character = list(alphabet_character_file.read().strip())
        alphabet_character_raw = ['START', '\xad']
        for item in alphabet_character:
            alphabet_character_raw.append(item)
        alphabet_character_raw.append('END')
        alphabet_character = alphabet_character_raw
        alp2num = {}
        for index, char in enumerate(alphabet_character):
            alp2num[char] = index
        alp2num_character = alp2num

    return alphabet_character

def tensor2str(tensor, alpha_path):
    alphabet = get_alphabet(alpha_path)
    string = ""
    for i in tensor:
        if i == (len(alphabet)-1):
            continue
        string += alphabet[i]
    return string