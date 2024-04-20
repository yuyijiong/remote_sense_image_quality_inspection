import os
from typing import Callable

import accelerate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration,GitForCausalLM,AutoTokenizer,AutoModelForCausalLM

kwargs = GradScalerKwargs()
accelerator = Accelerator(device_placement=True,mixed_precision="no",kwargs_handlers=[kwargs])
TQDM_INTERVAL = 1

# 训练loop，训练一个epoch
@accelerate.find_executable_batch_size(starting_batch_size=32)
def train_loop_fn(batch_size: int, dataset, model, optimizer, data_collator):
    model.train()
    #根据batchsize调整gradient_accumulation_steps
    accelerator.gradient_accumulation_steps =  max(32//batch_size,1)
    # 打印batch_size，红色字体
    print("\033[1;31mtrain_batch_size:", batch_size,
          'accumulation_steps',accelerator.gradient_accumulation_steps,"\033[0m")
    # 打印训练集长度，红色字体
    print("\033[1;31mtrain_dataset_length:", len(dataset), "\033[0m")
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=data_collator)
    # acclelrator包装
    data_loader = accelerator.prepare(data_loader)

    losses = []
    tk0 = tqdm(data_loader, total=len(data_loader), mininterval=TQDM_INTERVAL, desc='train')

    with accelerator.accumulate(model):
        with accelerator.autocast():
            for step, batch in enumerate(tk0):
                optimizer.zero_grad()

                try:
                    outputs = model(**batch)
                    loss = outputs[0]
                except:
                    # #打印出错的batch的input_ids的shape
                    # print('input_ids',batch['input_ids'].shape)
                    # #打印出错的batch的labels的shape和decoder_input_ids的shape
                    # print('labels',batch['labels'].shape,'decoder_input_ids',batch['decoder_input_ids'].shape)
                    raise

                accelerator.backward(loss)
                optimizer.step()

                # accelerator.backward(accelerator.scaler.scale(loss))
                # accelerator.scaler.step(optimizer)
                # accelerator.scaler.update()

                losses.append(loss.detach().cpu().item())
                # 每隔100个batch打印一次loss
                if step % TQDM_INTERVAL == 0:
                    tk0.set_postfix(loss=loss.detach().cpu().item())


    accelerator.clear()

    # 计算平均loss
    avg_loss = np.mean(losses)
    return avg_loss


# 验证loop，验证一个epoch
@accelerate.find_executable_batch_size(starting_batch_size=128)
def valid_loop_fn(batch_size: int, dataset: Dataset, model, data_collator: Callable = None) -> float:
    model.eval()
    losses = []
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=data_collator, drop_last=False)
    interval = len(data_loader) // 10 + 1
    # 打印数据集长度，红色字体
    print("\033[1;31mvalid_dataset_length:", len(dataset), "\033[0m")
    # acclelrator包装
    data_loader = accelerator.prepare(data_loader)
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader), mininterval=TQDM_INTERVAL, desc='valid')
        for step, batch in enumerate(tk0):
            with accelerator.autocast():
                outputs = model(**batch)
            loss = outputs[0]
            # 当前batch的长度,即batch的第一个value的shape[0]
            batch_len= list(batch.values())[0].shape[0]
            losses.append(loss.cpu().item() * batch_len)

    torch.cuda.empty_cache()
    accelerator.clear()

    # 计算平均loss
    avg_loss = np.sum(losses) / len(dataset)

    return avg_loss

# 测试loop，测试一个epoch
@accelerate.find_executable_batch_size(starting_batch_size=128)
def test_loop_fn_generate(batch_size: int, dataset, model: T5ForConditionalGeneration, tokenizer, data_collator,
                          max_len=256):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator,
                             drop_last=False)
    data_loader = accelerator.prepare(data_loader)
    # #如果模型不在cuda上
    # if model.device !='cuda':
    #     model.cuda()

    model.eval()
    generated_text_list = []
    target_text_list = []

    with torch.no_grad():
        with accelerator.autocast():
            for data in tqdm(data_loader, desc='Evaluate Generate', total=len(data_loader), mininterval=TQDM_INTERVAL):
                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                # 生成的结果必须包含'|'
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    min_length=3,
                    max_length=max_len,
                    max_new_tokens=None,
                    num_beams=4,
                    repetition_penalty=2.5,
                    length_penalty=1.5,
                    early_stopping=True,
                    no_repeat_ngram_size=4,
                    num_beam_groups=1,
                    force_words_ids=None,
                    bad_words_ids=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,

                )

                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                generated_text_list.extend(decoded_preds)

                # 检测dataset是否有label列
                if 'label' in dataset.column_names or 'labels' in dataset.column_names:
                    labels = data['labels'] if 'labels' in data else data['label']
                    # Replace -100 in the labels as we can't decode them
                    labels = labels.cpu().numpy()
                    labels[labels == -100] = tokenizer.pad_token_id
                    # Decode labels
                    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)  # 返回的是一个list，每个元素是一个字符串
                    target_text_list.extend(decoded_labels)

    return generated_text_list, target_text_list


# 测试loop，测试一个epoch
@accelerate.find_executable_batch_size(starting_batch_size=1)
def test_loop_fn_generate_vqa(batch_size: int, dataset, model: GitForCausalLM, tokenizer, data_collator,
                          max_len=64, num_return_sequences=1, do_sample=False):
    # 打印batch_size，红色字体
    print("\033[1;31mtest_batch_size:", batch_size,
          'accumulation_steps', accelerator.gradient_accumulation_steps, "\033[0m")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator,
                             drop_last=False)
    data_loader = accelerator.prepare(data_loader)

    model.eval()
    generated_text_list = []
    target_text_list = []

    with torch.no_grad():
        with accelerator.autocast():
            for batch in tqdm(data_loader, desc='Evaluate Generate', total=len(data_loader),
                             mininterval=float(os.environ.get('TQDM_INTERVAL'))):

                # 检测dataset是否有label列
                if 'labels' in batch:
                    labels = batch.pop('labels')
                    # Replace -100 in the labels as we can't decode them
                    labels = labels.cpu().numpy()
                    labels[labels == -100] = tokenizer.pad_token_id
                    # Decode labels
                    decoded_labels = tokenizer.batch_decode(labels,
                                                            skip_special_tokens=True)  # 返回的是一个list，每个元素是一个字符串
                    target_text_list.extend(decoded_labels)

                generated_ids = model.generate(
                    **batch,
                    min_length=3,
                    do_sample=do_sample,
                    max_length=max_len,
                    num_beams=1,
                    early_stopping=True,
                    num_return_sequences=num_return_sequences,
                )

                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                generated_text_list.extend(decoded_preds)

    return generated_text_list, target_text_list


# 测试loop，测试一个epoch,并收集所有输出
@accelerate.find_executable_batch_size(starting_batch_size=4)
def test_loop_fn(batch_size: int, dataset, model: GitForCausalLM, data_collator,**kwargs):
    # #如果model的参数里有类型为fp16的参数，那么accelerator的mixed_precision设为fp16
    # if torch.float16 in [str(param.type) for param in model.parameters()]:
    #     accelerator.mixed_precision = 'fp16'


    # 打印batch_size，红色字体
    print("\033[1;31mtest_batch_size:", batch_size,
          'accumulation_steps', accelerator.gradient_accumulation_steps, "\033[0m")

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator,
                             drop_last=False)
    data_loader = accelerator.prepare(data_loader)
    model=accelerator.prepare(model)
    model.eval()
    result_list=[]


    with torch.no_grad():
        with accelerator.autocast():
            for batch in tqdm(data_loader, desc='Get Output', total=len(data_loader),
                             mininterval=float(os.environ.get('TQDM_INTERVAL'))):

                # 检测batch是否有labels
                if 'labels' in batch:
                    batch.pop('labels')

                #获取输出
                outputs = model(**batch,**kwargs)

                #检查所有value的类型，如果是tensor，就将其转换为cpu。否则删除这个key
                for key,value in outputs.items():
                    if isinstance(value,torch.Tensor):
                        outputs[key]=value.cpu()
                    else:
                        outputs.pop(key)

                #收集输出
                result_list.append(outputs)



    #将所有输出拼接起来，即将每个元素中同一个key的value在第一维拼接起来
    result_dict={}
    for key in result_list[0].keys():
        result_dict[key]=torch.cat([result[key] for result in result_list],dim=0)
    #result_dict转换为和outputs一样的格式
    for key in outputs.keys():
        outputs[key]=result_dict[key]

    return outputs








