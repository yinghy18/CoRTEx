import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import faiss
import random
import string
import time
import pickle
import gc
import argparse
from InstructorEmbedding import INSTRUCTOR
from accelerate import Accelerator

batch_size = 256
#device = torch.device("cuda:0")
accelerator = Accelerator()
device = accelerator.device

def get_instructor_embed(phrase_list, m, batch_size, normalize=True, show_progress_bar=True):
    texts_with_instructions = []
    m.max_len = 96
    for phrase in phrase_list:
        texts_with_instructions.append(['Represent the meaning of the biomedical term for retrieval: ', phrase])
        #texts_with_instructions.append([phrase.split('Input: ')[0]+'Input: ',phrase.split('Input: ')[1]])
    accelerator = Accelerator()
    m = accelerator.prepare(m)
    m.eval()
    output = m.encode(texts_with_instructions, batch_size=batch_size, show_progress_bar =  show_progress_bar, normalize_embeddings=normalize)
    torch.cuda.empty_cache()
    return output

def get_KNN(embeddings, k):
    d = embeddings.shape[1]
    #res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(d)
    #gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    gpu_index.add(embeddings)
    print(gpu_index.ntotal)
    similarity, indices = gpu_index.search(embeddings.astype(np.float32), k)
    del gpu_index
    gc.collect()
    return similarity, indices

def find_new_index(indices_path, similarity_path, embedding_path, phrase2idx_path, ori_Instructor_path):
    print('start finding new index...')
    model= torch.load(ori_Instructor_path).to(device)
    print('start loading phrases...')
    with open(phrase2idx_path, 'rb') as f:
        phrase2idx = pickle.load(f)
    phrase_list = list(phrase2idx.keys())
    embeddings = get_instructor_embed(phrase_list, model, batch_size)
    del model
    torch.cuda.empty_cache()
    with open(embedding_path, 'wb') as f:
        np.save(f, embeddings)
    print('start knn')
    similarity, indices = get_KNN(embeddings, 30)
    with open(indices_path, 'wb') as f:
        np.save(f, indices)
    with open(similarity_path, 'wb') as f:
        np.save(f, similarity)
    print('done knn')
    return None


if __name__ == "__main__":
    print(torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Instructor_name",
        default="last_model.pth",
        type=str,
        help="Path to Instructor"
    )
    parser.add_argument(
        "--save_dir",
        default="./data/",
        type=str,
        help="output dir"
    )
    parser.add_argument(
        "--phrase2idx_path",
        default="./data/phrase2idx.pkl",
        type=str,
        help="Path to phrase2idx file"
    )
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    args.indices_path = os.path.join(args.save_dir, 'indices.npy')
    args.similarity_path = os.path.join(args.save_dir, 'similarity.npy')
    args.embedding_path = os.path.join(args.save_dir, 'embedding.npy')
    
    find_new_index(
        ori_Instructor_path=args.Instructor_name,
        indices_path=args.indices_path,
        similarity_path=args.similarity_path,
        embedding_path=args.embedding_path,
        phrase2idx_path=args.phrase2idx_path
    )


