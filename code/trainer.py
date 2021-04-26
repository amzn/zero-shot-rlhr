import os 
import sys 
import numpy as np 
from tqdm import tqdm 

import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup

import utils
from dataloader import DatasetForBert, DatasetForRL
from policy import Policy


def prepare_for_training(paras, model, amp=None):
    # define the optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': paras['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=paras['learning_rate'], eps=paras['adam_epsilon'])

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=paras['fp16_opt_level'])

    # multi-gpu training (should be after apex fp16 initialization)
    if paras['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if paras['local_rank'] != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[paras['local_rank']], output_device=paras['local_rank'], find_unused_parameters=True)

    return model, optimizer


def do_train(paras, training_features, model, tokenizer, logger):

    if paras['fp16']:
        from apex import amp
    else:
        amp = None

    # get the max training step if continue to train
    recover_step = utils.get_max_epoch_model(paras['cache_dir'])

    model.to(paras['device'])
    model, optimizer = prepare_for_training(paras, model, amp=amp)

    # define the total batch size
    if paras['n_gpu'] == 0 or paras['no_cuda']:
        per_node_train_batch_size = paras['per_gpu_train_batch_size'] * paras['gradient_accumulation_steps']
    else:
        per_node_train_batch_size = paras['per_gpu_train_batch_size'] * paras['n_gpu'] * paras['gradient_accumulation_steps']
        
    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if paras['local_rank'] != -1 else 1)
    global_step = recover_step if recover_step else 0

    # the total training steps
    if paras['num_training_steps'] == -1:
        paras['num_training_steps'] = int(paras['num_training_epochs'] * len(training_features) / train_batch_size)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=paras['num_warmup_steps'],
        num_training_steps=paras['num_training_steps'], last_epoch=-1)

    # dataset
    train_dataset = DatasetForBert(
        features=training_features,
        max_source_len=paras['max_source_seq_length'], 
        max_seq_len=paras['max_seq_length'], 
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, 
        pad_id=tokenizer.pad_token_id, mask_id=tokenizer.mask_token_id, 
        offset=train_batch_size * global_step, 
        num_training_instances=train_batch_size * paras['num_training_steps'],
        use_path=paras['use_path']
    )

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_features))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_features))
    logger.info("  Instantaneous batch size per GPU = %d", paras['per_gpu_train_batch_size'])
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", paras['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", paras['num_training_steps'])

    if paras['num_training_steps'] <= global_step:
        logger.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if paras['local_rank'] == -1 else DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // paras['gradient_accumulation_steps'],
            collate_fn=utils.batch_list_to_batch_tensors)

        train_iterator = tqdm(
            train_dataloader, initial=global_step,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=paras['local_rank'] not in [-1, 0])

        model.train()
        model.zero_grad()

        logging_loss = 0.0

        for step, batch in enumerate(train_iterator):
            batch = tuple(t.to(paras['device']) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if paras['mode'] != 'distilbert':
                inputs['token_type_ids'] = batch[2]
                
            outputs = model(**inputs)
            loss = outputs[0]
            if paras['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training

            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (loss.item(), scheduler.get_last_lr()[0]))

            if paras['gradient_accumulation_steps'] > 1:
                loss = loss / paras['gradient_accumulation_steps']
            
            if paras['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            logging_loss += loss.item()
            if (step + 1) % paras['gradient_accumulation_steps'] == 0:
                if paras['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), paras['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), paras['max_grad_norm'])

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if paras['local_rank'] in [-1, 0] and paras['logging_steps'] > 0 and global_step % paras['logging_steps'] == 0:
                    logger.info(" Step [%d ~ %d]: %.2f", global_step - paras['logging_steps'], global_step, logging_loss)
                    logging_loss = 0.0

                if paras['local_rank'] in [-1, 0] and paras['save_steps'] > 0 and \
                        (global_step % paras['save_steps'] == 0 or global_step == paras['num_training_steps']):

                    save_path = os.path.join(paras['cache_dir'], "ckpt-%d" % global_step)
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(save_path)
                    
                    logger.info("Saving model checkpoint %d into %s", global_step, save_path)


def do_rl_train(paras, training_data, model, tokenizer, logger):

    if paras['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    # get the max training step if continue to train
    recover_step = utils.get_max_epoch_model(paras['cache_dir'])

    model.to(paras['device'])
    model, optimizer = prepare_for_training(paras, model, amp=amp)

    # get the training batch size
    if paras['n_gpu'] == 0 or paras['no_cuda']:
        per_node_train_batch_size = paras['per_gpu_train_batch_size'] * paras['gradient_accumulation_steps']
    else:
        per_node_train_batch_size = paras['per_gpu_train_batch_size'] * paras['n_gpu'] * paras['gradient_accumulation_steps']
    

    train_batch_size = per_node_train_batch_size * (torch.distributed.get_world_size() if paras['local_rank'] != -1 else 1)
    global_step = recover_step if recover_step else 0

    if paras['num_training_steps'] == -1:
        paras['num_training_steps'] = int(paras['num_training_epochs'] * len(training_data['examples_id']) / train_batch_size)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=paras['num_warmup_steps'],
        num_training_steps=paras['num_training_steps'], last_epoch=-1)
    
    train_dataset = DatasetForRL(
        data=training_data,
        max_source_len=paras['max_source_seq_length'], 
        max_seq_len=paras['max_seq_length'], 
        cls_id=tokenizer.cls_token_id, sep_id=tokenizer.sep_token_id, 
        pad_id=tokenizer.pad_token_id, mask_id=tokenizer.mask_token_id, 
        node_sep_id=tokenizer.convert_tokens_to_ids("-"), offset=train_batch_size * global_step, 
        num_training_instances=train_batch_size * paras['num_training_steps'],
        use_path = paras['use_path']
    )
    policy = Policy(taxonomy_id=training_data['taxonomy_id'], path_max_length=paras['path_max_length'], right_node_reward=paras['pos_reward'])

    # Train!
    logger.info("  ***** Running training *****  *")
    logger.info("  Num examples = %d", len(training_data['examples_id']))
    logger.info("  Num Epochs = %.2f", len(train_dataset) / len(training_data['examples_id']))
    logger.info("  Instantaneous batch size per GPU = %d", paras['per_gpu_train_batch_size'])
    logger.info("  Batch size per node = %d", per_node_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", paras['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", paras['num_training_steps'])

    if paras['num_training_steps'] <= global_step:
        logger.info("Training is done. Please use a new dir or clean this dir!")
    else:
        # The training features are shuffled
        train_sampler = SequentialSampler(train_dataset) \
            if paras['local_rank'] == -1 else DistributedSampler(train_dataset, shuffle=False)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler,
            batch_size=per_node_train_batch_size // paras['gradient_accumulation_steps'],
            collate_fn=utils.init_rl_batch)

        train_iterator = tqdm(
            train_dataloader, initial=global_step,
            desc="Iter (loss=X.XXX, lr=X.XXXXXXX)", disable=paras['local_rank'] not in [-1, 0])
        
        model.train()
        model.zero_grad()

        logging_loss = 0.0

        for step, raw_batch in enumerate(train_iterator):
            text = raw_batch[0]
            category = raw_batch[1]
            path = raw_batch[2]
            batch_size = len(text)

            text = raw_batch[0]
            current_node = np.asarray(raw_batch[3])
            batch_index = np.asarray([i for i in range(len(text))])
            batch_size = len(batch_index)
            policy.init_state(current_node, mode='random')
            while True:
                # acquire the action space
                text, current_node, batch_index = policy.get_children(text, current_node, batch_index)
                if len(current_node) == 0:
                    break
                batch = train_dataset.build_batch(text, current_node)
                batch = tuple(t.to(paras['device']) for t in batch)
                inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]}
                if paras['mode'] != 'distilbert' and paras['mode'] != 'distilbert-rl':
                    inputs['token_type_ids'] = batch[2]
                outputs = model(**inputs)
                outputs_prob = outputs[0]
                prob = F.log_softmax(outputs_prob, dim=1)

                # sampel actions according to the prob distribution
                index = policy.get_random_action(torch.exp(prob), batch_index, batch_size, paras['K'])
                # update the state by the sampled action
                policy.update_state(index, prob, current_node, batch_index)
                # acquire the next state
                text, current_node, batch_index = policy.select_current_node(index, text, current_node, batch_index)
            random_reward = policy.get_reward(policy.random_state, category, path)
            
            policy_loss = sum(random_reward)
            policy_loss = policy_loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            train_iterator.set_description('Iter (loss=%5.3f) lr=%9.7f' % (policy_loss.item(), scheduler.get_last_lr()[0]))

            if paras['gradient_accumulation_steps'] > 1:
                policy_loss = policy_loss / paras['gradient_accumulation_steps']
            
            if paras['fp16']:
                with amp.scale_loss(policy_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                policy_loss.backward()

            logging_loss += policy_loss.item()
            if (step + 1) % paras['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), paras['max_grad_norm'])

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if paras['local_rank'] in [-1, 0] and paras['logging_steps'] > 0 and global_step % paras['logging_steps'] == 0:
                    logger.info(" Step [%d ~ %d]: %.2f", global_step - paras['logging_steps'], global_step, logging_loss)
                    logging_loss = 0.0

                if paras['local_rank'] in [-1, 0] and paras['save_steps'] > 0 and \
                        (global_step % paras['save_steps'] == 0 or global_step == paras['num_training_steps']):

                    save_path = os.path.join(paras['cache_dir'], "ckpt-%d" % global_step)
                    os.makedirs(save_path, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(save_path)
                    
                    logger.info("Saving model checkpoint %d into %s", global_step, save_path)



            

                
            