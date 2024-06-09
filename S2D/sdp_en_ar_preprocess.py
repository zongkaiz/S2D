import json
from argparse import ArgumentParser, Namespace

import networkx as nx
import numpy as np
import stanza
import torch
import tqdm
from tqdm import tqdm
from transformers import MT5Tokenizer
import hanlp#k
from hanlp_common.conll import CoNLLSentence#k
from data import GenDataset

upos2id = {
    'PAD': 0, 'INTJ': 1, 'PART': 2, 'NOUN': 3, 'PUNCT': 4, 'SYM': 5,
    'NUM': 6, 'PROPN': 7, 'ADJ': 8, 'PRON': 9, 'X': 10,
    'SCONJ': 11, 'ADV': 12, 'DET': 13, 'CCONJ': 14, 'ADP': 15,
    'AUX': 16, 'VERB': 17
}


def process_train(dataset, tokenizer, upos2id, deprel2id=None, language='en', max_token_len=350, max_prefix_len=5):
    sdp_max_prefix_len=20#k
    pos = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)#k
    sdp = hanlp.load(hanlp.pretrained.sdp.SEMEVAL15_PSD_BIAFFINE_EN)#k

    nlp = stanza.Pipeline(language, tokenize_pretokenized=True)
    for data in tqdm(dataset):
        words = data['info']['tokens']
        passage = data['info']['passage']
        doc = nlp([words])
        pos_result=pos(words)#k
        sent = [(w, p) for w, p in zip(words, pos_result)]#k
        sdp_doc = sdp(sent)#k
        pos_tags = []
        heads = []
        deprels = []
        words = []
        for sentence in doc.sentences:
            for w in sentence.words:
                pos_tags.append(w.upos)#词性
                heads.append(w.head - 1)
                deprels.append(w.deprel)#依存关系标签（dependency relation label）。在自然语言处理（NLP）和语言学中，依存关系用于描述句子中词汇之间的语法关系，而 deprel 则是这种关系的标签。
                words.append(w.text)
        start_ids = []
        end_ids = []
        word_pieces = []
        pos_ids = []
        for word, pos_tag in zip(words, pos_tags):
            start_ids.append(len(word_pieces))
            word_piece = tokenizer.tokenize(word)#homeland被分为‘home’,'##land'
            end_ids.append(len(word_pieces) + len(word_piece))
            word_pieces.extend(word_piece)
            pos_ids.extend([upos2id[pos_tag]] * len(word_piece))

        input_ids = tokenizer.convert_tokens_to_ids(word_pieces)
        input_ids = input_ids[:max_token_len]
        attention_mask = [1] * len(input_ids)
        pos_ids = pos_ids[:max_token_len]
        input_ids += [0] * (max_token_len - len(input_ids))
        attention_mask += [0] * (max_token_len - len(attention_mask))
        pos_ids += [0] * (max_token_len - len(pos_ids))
        syntax_attention_mask = torch.zeros(max_token_len, max_token_len)

        # 建图
        g = nx.Graph()
        sdp_g=nx.Graph()#k

        for i in range (len(sdp_doc)):#k
            sdp_g.add_edge(sdp_doc[i]["id"], sdp_doc[i]['head'][0])#k
            sdp_g.add_edge(sdp_doc[i]['head'][0], sdp_doc[i]["id"])#k
            
        for sentence in doc.sentences:
            for dep in sentence.dependencies:
                g.add_edge(dep[0].id, dep[2].id)
                g.add_edge(dep[2].id, dep[0].id)

        for start, end in zip(start_ids, end_ids):
            for i in range(start, end):
                for j in range(start, end):
                    syntax_attention_mask[i][j] = 1
                    syntax_attention_mask[j][i] = 1

        for i in range(1, 1 + len(words)):
            for j in range(1, 1 + len(words)):
                if i != j:
                    dis = nx.shortest_path_length(g, i, j)
                    for m in range(start_ids[i - 1], end_ids[i - 1]):
                        for n in range(start_ids[j - 1], end_ids[j - 1]):
                            syntax_attention_mask[m][n] = dis
                            syntax_attention_mask[n][m] = dis

        trigger_span = data['info']['trigger span']
        root = trigger_span[0] + 1

        # 广度优先遍历
        sdp_bfs_result = list(nx.bfs_tree(sdp_g, 0))#k
        sdp_prefix_ids = []#k
        for i in sdp_bfs_result:#k
            if i != 0:#k
                for j in range(start_ids[i - 1], end_ids[i - 1]):#k
                    sdp_prefix_ids.append(j)#k
        sdp_prefix_ids = sdp_prefix_ids + [start_ids[trigger_span[0]]] * (sdp_max_prefix_len - len(sdp_prefix_ids))#k
        sdp_prefix_ids = sdp_prefix_ids[:sdp_max_prefix_len]#k   sdp_max_prefix_len在这个函数上面和下面的process_test函数同样位置先定义了=10  但应该需要在函数变量和配置文件中增加一个这个变量

        bfs_result = list(nx.bfs_tree(g, root))
        prefix_ids = []
        for i in bfs_result:
            if i != 0:
                for j in range(start_ids[i - 1], end_ids[i - 1]):
                    prefix_ids.append(j)
        prefix_ids = prefix_ids + [start_ids[trigger_span[0]]] * (max_prefix_len - len(prefix_ids))
        #prefix_ids = prefix_ids[:max_prefix_len]
        prefix_ids=prefix_ids[:(max_prefix_len-sdp_max_prefix_len)]#k
        sdp_index=max_prefix_len-sdp_max_prefix_len#k
        prefix_ids[sdp_index:]=sdp_prefix_ids#k
        
        input_ids = torch.LongTensor(input_ids)
        pos_ids = torch.LongTensor(pos_ids)
        prefix_ids = torch.LongTensor(prefix_ids)
        attention_mask = torch.LongTensor(attention_mask)

        data['encoder'] = {}
        data['encoder']['input_ids'] = input_ids
        data['encoder']['pos_ids'] = pos_ids
        data['encoder']['attention_mask'] = attention_mask
        data['encoder']['syntax_mask'] = syntax_attention_mask
        data['encoder']['prefix_ids'] = prefix_ids


def process_test(dataset, tokenizer, upos2id, deprel2id=None, language='en', max_token_len=350, max_prefix_len=5):
    sdp_max_prefix_len=20#k
    sdp = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_MMINILMV2L12)
    
    nlp = stanza.Pipeline(language, tokenize_pretokenized=True)
    for data in tqdm(dataset):
        words = data.tokens
        doc = nlp([words])
        ar_sdp_doc = sdp(words, tasks='sdp')#k
        sdp_doc=ar_sdp_doc["sdp/dm"]#k
        pos_tags = []
        heads = []
        deprels = []
        words = []
        for sentence in doc.sentences:
            for w in sentence.words:
                pos_tags.append(w.upos)
                heads.append(w.head - 1)
                deprels.append(w.deprel)
                words.append(w.text)
        start_ids = []
        end_ids = []
        word_pieces = []
        pos_ids = []
        for word, pos_tag in zip(words, pos_tags):
            start_ids.append(len(word_pieces))
            word_piece = tokenizer.tokenize(word)
            end_ids.append(len(word_pieces) + len(word_piece))
            word_pieces.extend(word_piece)
            pos_ids.extend([upos2id[pos_tag]] * len(word_piece))

        input_ids = tokenizer.convert_tokens_to_ids(word_pieces)
        input_ids = input_ids[:max_token_len]
        attention_mask = [1] * len(input_ids)
        pos_ids = pos_ids[:max_token_len]
        input_ids += [0] * (max_token_len - len(input_ids))
        attention_mask += [0] * (max_token_len - len(attention_mask))
        pos_ids += [0] * (max_token_len - len(pos_ids))
        syntax_attention_mask = torch.zeros(max_token_len, max_token_len)

        # 建图
        g = nx.Graph()
        sdp_g=nx.Graph()#k
        for i in range (len(sdp_doc)):#k
            for j in range(len(sdp_doc[i])):
                if not sdp_doc[i][j]:
                    sdp_g.add_edge(i+1, 0)#k
                    sdp_g.add_edge(0,i+1)#k
                else:
                    for k in range(len(sdp_doc[i][j])):
                        sdp_g.add_edge(i+1, sdp_doc[i][j][k][0])#k
                        sdp_g.add_edge(sdp_doc[i][j][k][0],i+1)#k
             
        for sentence in doc.sentences:
            for dep in sentence.dependencies:
                g.add_edge(dep[0].id, dep[2].id)
                g.add_edge(dep[2].id, dep[0].id)

        for start, end in zip(start_ids, end_ids):
            for i in range(start, end):
                for j in range(start, end):
                    syntax_attention_mask[i][j] = 1
                    syntax_attention_mask[j][i] = 1

        for i in range(1, 1 + len(words)):
            for j in range(1, 1 + len(words)):
                if i != j:
                    dis = nx.shortest_path_length(g, i, j)
                    for m in range(start_ids[i - 1], end_ids[i - 1]):
                        for n in range(start_ids[j - 1], end_ids[j - 1]):
                            syntax_attention_mask[m][n] = dis
                            syntax_attention_mask[n][m] = dis

        triggers = data.triggers
        for trigger in triggers:
            prefix_ids = []
            root = trigger[0] + 1

            # 广度优先遍历
            sdp_bfs_result = list(nx.bfs_tree(sdp_g, 0))#k
            sdp_prefix_ids = []#k
            for i in sdp_bfs_result:#k
                if i != 0:#k
                    for j in range(start_ids[i - 1], end_ids[i - 1]):#k
                        sdp_prefix_ids.append(j)#k
            sdp_prefix_ids = sdp_prefix_ids + [start_ids[trigger[0]]] * (sdp_max_prefix_len - len(sdp_prefix_ids))#k
            sdp_prefix_ids = sdp_prefix_ids[:sdp_max_prefix_len]#k

            bfs_result = list(nx.bfs_tree(g, root))
            for i in bfs_result:
                if i != 0:
                    for j in range(start_ids[i - 1], end_ids[i - 1]):
                        prefix_ids.append(j)

            prefix_ids = prefix_ids + [start_ids[trigger[0]]] * (max_prefix_len - len(prefix_ids))
            #prefix_ids = prefix_ids[:max_prefix_len]
            prefix_ids=prefix_ids[:(max_prefix_len-sdp_max_prefix_len)]#k
            sdp_index=max_prefix_len-sdp_max_prefix_len#k
            prefix_ids[sdp_index:]=sdp_prefix_ids#k

            pos_ids = torch.LongTensor(pos_ids)
            input_ids = torch.LongTensor(input_ids)
            prefix_ids = torch.LongTensor(prefix_ids)
            attention_mask = torch.LongTensor(attention_mask)

            data.encoder_pos_ids.append(pos_ids)
            data.encoder_input_ids.append(input_ids)
            data.encoder_attention_mask.append(attention_mask)
            data.prefix_ids.append(prefix_ids)
            data.encoder_syntax_mask.append(syntax_attention_mask)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default="../config/config_ace05_mT5copy-base_en_prompt.json")
    args = parser.parse_args()
    with open(args.config) as fp:
        config = json.load(fp)
    config.update(args.__dict__)
    config = Namespace(**config)

    # import template file
    if config.dataset == "ace05":
        from template_generate_ace import IN_SEP, ROLE_LIST, NO_ROLE, AND

        TEMP_FILE = "template_generate_ace"
    elif config.dataset == "ere":
        from template_generate_ere import IN_SEP, ROLE_LIST, NO_ROLE, AND

        TEMP_FILE = "template_generate_ere"
    else:
        raise NotImplementedError

    # fix random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.enabled = False

    # set GPU device
    torch.cuda.set_device(config.gpu_device)

    # tokenizer
    if config.model_name.startswith("google/mt5-"):
        tokenizer = MT5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    elif config.model_name.startswith("copy+google/mt5-"):
        model_name = config.model_name.split('copy+', 1)[1]
        tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)
    else:
        raise NotImplementedError

    special_tokens = []
    sep_tokens = []
    if "triggerword" in config.input_style:
        sep_tokens += [IN_SEP["triggerword"]]
    if "template" in config.input_style:
        sep_tokens += [IN_SEP["template"]]
    if "argument:roletype" in config.output_style:
        special_tokens += [f"<--{r}-->" for r in ROLE_LIST]
        special_tokens += [f"</--{r}-->" for r in ROLE_LIST]
        special_tokens += [NO_ROLE, AND]
    tokenizer.add_tokens(sep_tokens + special_tokens)

    upos2id = {
        'PAD': 0, 'INTJ': 1, 'PART': 2, 'NOUN': 3, 'PUNCT': 4, 'SYM': 5,
        'NUM': 6, 'PROPN': 7, 'ADJ': 8, 'PRON': 9, 'X': 10,
        'SCONJ': 11, 'ADV': 12, 'DET': 13, 'CCONJ': 14, 'ADP': 15,
        'AUX': 16, 'VERB': 17
    }
    train_set = GenDataset(tokenizer, sep_tokens, config.max_length, config.train_finetune_file,
                           config.max_output_length)
    process_train(train_set, tokenizer, upos2id, language=config.alias)
