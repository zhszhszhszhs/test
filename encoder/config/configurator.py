import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'modelconf')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    model_config_path = os.path.join(CONFIG_DIR, '{}.yml'.format(model_name))
    if not os.path.exists(model_config_path):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open(model_config_path, encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed

        # semantic embeddings
        data_dir = os.path.join(DATA_DIR, configs['data']['name'])
        usrprf_embeds_path = os.path.join(data_dir, 'usr_emb_np.pkl')
        itmprf_embeds_path = os.path.join(data_dir, 'itm_emb_np.pkl')
        with open(usrprf_embeds_path, 'rb') as f:
            configs['usrprf_embeds'] = pickle.load(f)
        with open(itmprf_embeds_path, 'rb') as f:
            configs['itmprf_embeds'] = pickle.load(f)
        #
        # # LLM-derived semantic embeddings
        usrint_embeds_path = os.path.join(data_dir, 'user_intent_emb_3.pkl')
        itmint_embeds_path = os.path.join(data_dir, 'item_intent_emb_3.pkl')
        with open(usrint_embeds_path, 'rb') as f:
            configs['user_semantic_embeds'] = pickle.load(f)
        with open(itmint_embeds_path, 'rb') as f:
            configs['item_semantic_embeds'] = pickle.load(f)
        configs['usrint_embeds'] = configs['user_semantic_embeds']
        configs['itmint_embeds'] = configs['item_semantic_embeds']

        if 'semantic_intent_prototype_file' in configs['model']:
            prototype_path = os.path.join(data_dir, configs['model']['semantic_intent_prototype_file'])
            configs['semantic_intent_prototypes'] = np.load(prototype_path, allow_pickle=False)

        return configs

configs = parse_configure()
