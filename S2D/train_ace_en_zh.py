import json
import logging
import os
import pprint
import sys
import time
from argparse import ArgumentParser, Namespace
#1
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import MT5Tokenizer, AdamW, get_linear_schedule_with_warmup, BertTokenizer

from data import EEDataset, GenDataset
from model import Prefix_fn_cls
from prefix_tuning import Model
# configuration
#from preprocess import process_train, process_test
from sdp_ar_preprocess import process_train, process_test
from utils import Summarizer, cal_scores, get_span_idxs, get_span_idxs_zh

parser = ArgumentParser()
parser.add_argument('--config', default='/media/h3c/users/zongkai/LAPIN/config-large/config_ace05_mT5copy-base_ar_ar_prompt_distance.json')
parser.add_argument('--constrained_decode', default=True, action='store_true')
parser.add_argument('--beam', type=int, default=4)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)
config.update(args.__dict__)
config = Namespace(**config)#将字典中的键值对转换为命名空间中的属性和属性值。这通常用于将配置文件中的参数加载到命名空间中，以便在代码中更方便地访问和使用这些参数。

# import template file
if config.dataset == "ace05":
    from template_generate_ace import event_template_generator, IN_SEP, ROLE_LIST, NO_ROLE, AND

    TEMP_FILE = "template_generate_ace"
elif config.dataset == "ere":
    from template_generate_ere import event_template_generator, IN_SEP, ROLE_LIST, NO_ROLE, AND

    TEMP_FILE = "template_generate_ere"
else:
    raise NotImplementedError

# fix random seed
np.random.seed(config.seed)
torch.manual_seed(config.seed)#设置随机数种子，确保生成的随机数是可复现的。
torch.backends.cudnn.enabled = False#在某些情况下，禁用 cuDNN（NVIDIA 提供的一个深度学习库，用于加速深度神经网络的训练和推理。PyTorch使用它来提高卷积操作等的性能。） 可能会导致性能下降，因为计算将不再受到 cuDNN 提供的优化。但在某些特殊情况下，例如调试或需要确保结果一致性时，禁用 cuDNN 可能是一个合理的选择。

# set GPU device
torch.cuda.set_device(config.gpu_device)

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]',
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

upos2id = {
    'PAD': 0, 'INTJ': 1, 'PART': 2, 'NOUN': 3, 'PUNCT': 4, 'SYM': 5,
    'NUM': 6, 'PROPN': 7, 'ADJ': 8, 'PRON': 9, 'X': 10,
    'SCONJ': 11, 'ADV': 12, 'DET': 13, 'CCONJ': 14, 'ADP': 15,
    'AUX': 16, 'VERB': 17
}


# check valid styles
assert np.all([style in ['triggerword', 'template'] for style in config.input_style])
assert np.all([style in ['argument:roletype'] for style in config.output_style])

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
# best_model_path = os.path.join(output_dir, 'best_model.mdl')
# dev_prediction_path = os.path.join(output_dir, 'pred.dev.json')
# test_prediction_path = os.path.join(output_dir, 'pred.test.json')

# tokenizer
if config.model_name.startswith("google/mt5-"):#k注释
    tokenizer = MT5Tokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)#k注释
    #tokenizer = MT5Tokenizer.from_pretrained(config.model_name)#k到时候上下都注释完之后这个把前面缩进去掉
elif config.model_name.startswith("copy+google/mt5-"):#k注释
    model_name = config.model_name.split('copy+', 1)[1]#k注释
    tokenizer = MT5Tokenizer.from_pretrained(model_name, cache_dir=config.cache_dir)#k注释
else:#k注释
    raise NotImplementedError#k注释

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
bert_tokenizer = BertTokenizer.from_pretrained('/media/h3c/users/zongkai/LAPIN/bert-base-multilingual-cased')

# load data
train_set = GenDataset(tokenizer, sep_tokens, config.max_length, config.train_finetune_file, config.max_output_length)
process_train(train_set, bert_tokenizer, upos2id, language=config.alias, max_prefix_len=30)

test_set = EEDataset(tokenizer, config.test_file, max_length=config.max_length)
process_test(test_set, bert_tokenizer, upos2id, language=config.alias_test, max_prefix_len=30)

train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)
with open(config.vocab_file_test) as f:
    vocab_test = json.load(f)


model = Model(config, tokenizer)
model.cuda(device=config.gpu_device)

# optimizer
param_groups = [{'params': model.parameters(), 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)
schedule = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=train_batch_num * config.warmup_epoch,
                                           num_training_steps=train_batch_num * config.max_epoch)

# start training
logger.info("Start training ...")
summarizer_step = 0
best_test_epoch = -1
best_test_scores = {
    'arg_id': (0.0, 0.0, 0.0),
    'arg_cls': (0.0, 0.0, 0.0)
}
for epoch in range(1, config.max_epoch + 1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")

    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(
            DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step,
                       shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):

        # forard model
        loss = model(batch)

        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1

        loss = loss * (1 / config.accumulate_step)
        loss.backward()

        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()

    progress.close()

    # eval test set
    progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test {}'.format(epoch))
    model.eval()
    test_gold_triggers, test_gold_roles, test_pred_roles = [], [], []
    test_pred_wnd_ids, test_gold_outputs, test_pred_outputs, test_inputs = [], [], [], []
    for batch in DataLoader(test_set, batch_size=config.eval_batch_size, shuffle=False, collate_fn=test_set.collate_fn):
        progress.update(1)
        batch_pred_roles = [[] for _ in range(config.eval_batch_size)]
        batch_pred_outputs = [[] for _ in range(config.eval_batch_size)]
        batch_gold_outputs = [[] for _ in range(config.eval_batch_size)]
        batch_inputs = [[] for _ in range(config.eval_batch_size)]
        batch_event_templates = []
        for tokens, triggers, roles, encoder_pos_ids, encoder_input_ids, encoder_attention_mask, prefix_ids, encoder_syntax_mask in zip(
                batch.tokens, batch.triggers, batch.roles, batch.encoder_pos_ids,
                batch.encoder_input_ids, batch.encoder_attention_mask, batch.prefix_ids, batch.encoder_syntax_mask):
            batch_event_templates.append(
                event_template_generator(tokens, triggers, roles, config.input_style, config.output_style, vocab_test,
                                         config.lang_test, encoder_pos_ids, encoder_input_ids,
                                         encoder_attention_mask, prefix_ids, encoder_syntax_mask))

        eae_inputs, eae_gold_outputs, eae_events, eae_bids = [], [], [], []
        encoder_input_ids, encoder_attention_mask, encoder_pos_ids, encoder_syntax_masks, prefix_ids = [], [], [], [], []
        for i, event_temp in enumerate(batch_event_templates):
            for data in event_temp.get_training_data():
                eae_inputs.append(data[0])
                eae_gold_outputs.append(data[1])
                eae_events.append(data[2])
                eae_bids.append(i)
                batch_inputs[i].append(data[0])
                if 'encoder_input_ids' in data[2]:
                    encoder_input_ids.append(data[2]['encoder_input_ids'])
                    encoder_attention_mask.append(data[2]['encoder_attention_mask'])
                    encoder_pos_ids.append(data[2]['encoder_pos_ids'])
                    encoder_syntax_masks.append(data[2]['encoder_syntax_mask'])
                    prefix_ids.append(data[2]['prefix_ids'])
        if len(eae_inputs) > 0:
            eae_inputs = tokenizer(eae_inputs, return_tensors='pt', padding=True, max_length=config.max_length + 2)
            enc_idxs = eae_inputs['input_ids']
            enc_idxs = enc_idxs.cuda()
            enc_attn = eae_inputs['attention_mask'].cuda()
            encoder_input_ids = torch.stack(encoder_input_ids, dim=0)
            encoder_attention_mask = torch.stack(encoder_attention_mask, dim=0)
            encoder_pos_ids = torch.stack(encoder_pos_ids, dim=0)
            encoder_syntax_masks = torch.stack(encoder_syntax_masks, dim=0)
            prefix_ids = torch.stack(prefix_ids, dim=0)
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_attention_mask = encoder_attention_mask.cuda()
            encoder_pos_ids = encoder_pos_ids.cuda()
            prefix_ids = prefix_ids.cuda()
            encoder_syntax_masks = encoder_syntax_masks.cuda()

            if config.beam_size == 1:
                model.pretrain_model._cache_input_ids = enc_idxs
            else:
                expanded_return_idx = (
                    torch.arange(enc_idxs.shape[0]).view(-1, 1).repeat(1, config.beam_size).view(-1).to(
                        enc_idxs.device))
                input_ids = enc_idxs.index_select(0, expanded_return_idx)
                model.pretrain_model._cache_input_ids = input_ids

            # inference
            bsz = enc_idxs.shape[0]

            # Encode description.
            description_representation = model.get_description_representation(batch)

            # Encode knowledge.

            knowledge_representation = model.get_knowledge_representation(encoder_input_ids, encoder_attention_mask,
                                                                          encoder_pos_ids, encoder_syntax_masks,
                                                                          prefix_ids)
            past_prompt = model.get_prompt(
                bsz=bsz, sample_size=config.beam_size, description=description_representation,
                knowledge=knowledge_representation,
            )

            if args.constrained_decode:
                prefix_fn_obj = Prefix_fn_cls(tokenizer, ["[and]"], enc_idxs)
                outputs = model.pretrain_model.generate(input_ids=enc_idxs, attention_mask=enc_attn,
                                                        past_prompt=past_prompt,
                                                        use_cache=True,
                                                        num_beams=config.beam_size,
                                                        max_length=config.max_output_length,
                                                        forced_bos_token_id=None,
                                                        prefix_allowed_tokens_fn=lambda batch_id,
                                                                                        sent: prefix_fn_obj.get(
                                                            batch_id, sent))
            else:

                outputs = model.pretrain_model.generate(
                    input_ids=enc_idxs,
                    attention_mask=enc_attn,
                    past_prompt=past_prompt,
                    use_cache=True,
                    num_beams=config.beam_size,
                    max_length=config.max_output_length,
                    forced_bos_token_id=None)

            # decode outputs
            eae_pred_outputs = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                for output in outputs]

            # extract argument roles from the generated outputs
            for p_text, g_text, info, bid in zip(eae_pred_outputs, eae_gold_outputs, eae_events, eae_bids):
                theclass = getattr(sys.modules[TEMP_FILE], info['event type'].replace(':', '_').replace('-', '_'),
                                   False)
                assert theclass
                template = theclass(config.input_style, config.output_style, info['tokens'], info['event type'],
                                    config.lang_test, info)
                pred_object = template.decode(p_text)
                for span, role_type, _ in pred_object:
                    # convert the predicted span to the offsets in the passage
                    # Chinese uses a different function since there is no space between Chenise characters
                    if config.lang_test == "chinese":
                        sid, eid = get_span_idxs_zh(batch.tokens[bid], span, trigger_span=info['trigger span'])
                    else:
                        sid, eid = get_span_idxs(batch.piece_idxs[bid], batch.token_start_idxs[bid], span, tokenizer,
                                                 trigger_span=info['trigger span'])

                    if sid == -1:
                        continue
                    batch_pred_roles[bid].append(
                        ((info['trigger span'] + (info['event type'],)), (sid, eid, role_type)))
                batch_gold_outputs[bid].append(g_text)
                batch_pred_outputs[bid].append(p_text)

        batch_pred_roles = [sorted(set(role)) for role in batch_pred_roles]
        test_gold_triggers.extend(batch.triggers)
        test_gold_roles.extend(batch.roles)
        test_pred_roles.extend(batch_pred_roles)
        test_pred_wnd_ids.extend(batch.wnd_ids)
        test_gold_outputs.extend(batch_gold_outputs)
        test_pred_outputs.extend(batch_pred_outputs)
        test_inputs.extend(batch_inputs)
    progress.close()

    # calculate scores
    test_scores = cal_scores(test_gold_roles, test_pred_roles)

    print("---------------------------------------------------------------------")
    print('Role I     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        test_scores['arg_id'][3] * 100.0, test_scores['arg_id'][2], test_scores['arg_id'][1],
        test_scores['arg_id'][4] * 100.0, test_scores['arg_id'][2], test_scores['arg_id'][0],
        test_scores['arg_id'][5] * 100.0))
    print('Role C     - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        test_scores['arg_cls'][3] * 100.0, test_scores['arg_cls'][2], test_scores['arg_cls'][1],
        test_scores['arg_cls'][4] * 100.0, test_scores['arg_cls'][2], test_scores['arg_cls'][0],
        test_scores['arg_cls'][5] * 100.0))
    print("---------------------------------------------------------------------")
    if test_scores['arg_cls'][5] * 100.0 >= best_test_scores['arg_cls'][2]:
        best_test_scores['arg_id'] = (
            test_scores['arg_id'][3] * 100.0, test_scores['arg_id'][4] * 100.0, test_scores['arg_id'][5] * 100.0)
        best_test_scores['arg_cls'] = (
            test_scores['arg_cls'][3] * 100.0, test_scores['arg_cls'][4] * 100.0, test_scores['arg_cls'][5] * 100.0)
        best_test_epoch = epoch
    logger.info({"best_epoch": best_test_epoch, "best_scores": best_test_scores})

    # write outputs
    #test_outputs = {}
    #for (test_pred_wnd_id, test_gold_trigger, test_gold_role, test_pred_role, test_gold_output, test_pred_output, test_input) in zip(
    #    test_pred_wnd_ids, test_gold_triggers, test_gold_roles, test_pred_roles, test_gold_outputs, test_pred_outputs, test_inputs):
    #    test_outputs[test_pred_wnd_id] = {
     #       "input": test_input, 
    #        "triggers": test_gold_trigger,
    #        "gold_roles": test_gold_role,
    #        "pred_roles": test_pred_role,
    #        "gold_text": test_gold_output,
     #       "pred_text": test_pred_output,
    #    }
    #new_file_name = 'test.pred' + str(epoch) + '.json'
    #with open(os.path.join(output_dir, new_file_name), 'w') as fp:
    #    json.dump(test_outputs, fp, indent=2)

logger.info(log_path)
logger.info("Done!")
