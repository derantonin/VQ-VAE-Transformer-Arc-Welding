import train_reconstruction_embedding
import train_classification_model

import logging as log
import argparse
import torch
import matplotlib
import torch.nn.functional as F
from torchmetrics.functional import accuracy, f1_score as f1
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image
from dataloader.asimow_dataloader import DataSplitId, ASIMoWDataModule, load_npy_data, ASIMoWDataLoader
from utils import get_latent_dataloader, print_training_input_shape
from model.mlp import MLP
from dataloader.utils import get_val_test_ids
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from model.vq_vae import VectorQuantizedVAE
from model.vq_vae_patch_embedd import VQVAEPatch

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

def get_model(model_name: str, file_name: str):
    model_path = f"./model_checkpoints/{model_name}/{file_name}"
    if os.path.exists(model_path):
        model_name = 'VQ VAE'
        print("model exists, no download needed")
    else:
        print("model missing")
        return None, None
    #model_name, model_path = get_pretrained_vqvae_models("vq_vae_model")
    vq_vae_dict = torch.load(model_path, map_location=torch.device('cpu'))
    print(vq_vae_dict['hyper_parameters'])
    vq_vae_dict['hyper_parameters'].pop('logger')
    model = VectorQuantizedVAE(**vq_vae_dict['hyper_parameters'])
    model.load_state_dict(vq_vae_dict['state_dict'])

    return model, vq_vae_dict['hyper_parameters']

def get_model_patch(model_name: str, file_name: str):
    model_path = f"./model_checkpoints/{model_name}/{file_name}"
    if os.path.exists(model_path):
        model_name = 'VQ VAE Patch'
        print("model exists, no download needed")
    else:
        print("model missing")
        return None, None
    #model_name, model_path = get_pretrained_vqvae_models("vq_vae_model")
    vq_vae_dict = torch.load(model_path, map_location=torch.device('cpu'))
    print(vq_vae_dict['hyper_parameters'])
    # vq_vae_dict['hyper_parameters'].pop('wandb_logger')
    model = VQVAEPatch(**vq_vae_dict['hyper_parameters'])
    model.load_state_dict(vq_vae_dict['state_dict'])

    return model, vq_vae_dict['hyper_parameters']



def get_models_and_files(mode: str):
    if mode not in ["VQ-VAE", "VQ-VAE-Patch", "VQ-VAE-Yannik-Patch"]:
        raise ValueError("mode must be one of 'VQ-VAE', 'VQ-VAE-Patch', 'VQ-VAE-Yannik-Patch'")

    if mode == "VQ-VAE":
        mlp_path = "MLPs/my_trained_mlp.ckpt"
        vqvae_path = "VQ-VAE-asimow-best.ckpt"
        my_trained_mlp = MLP(input_size=26, output_size=2, in_dim=32, hidden_sizes=512)
        my_trained_mlp.load_state_dict(torch.load(mlp_path))
        my_trained_mlp.eval()
        model, hparams = get_model("VQ-VAE", vqvae_path)
        model.eval()
        codebook = torch.round(model.vector_quantization.embedding.weight.data, decimals=3)
        complete_q_tensor = torch.round(torch.tensor(np.load("created_files/q_emb_v1.npy"), dtype=torch.float32), decimals=3)
        complete_q_indices = np.load("created_files/q_ind_v1.npy")
    elif mode == "VQ-VAE-Patch":
        mlp_path = "MLPs/my_trained_mlp_on_patch_v1.ckpt"
        vqvae_path = "VQ-VAE-Patch-best-v1.ckpt"
        my_trained_mlp = MLP(input_size=16, output_size=2, in_dim=16, hidden_sizes=512)
        # my_trained_mlp = None
        my_trained_mlp.load_state_dict(torch.load(mlp_path))
        my_trained_mlp.eval()
        model, hparams = get_model_patch("VQ-VAE-Patch", vqvae_path)
        model.eval()
        codebook=None
        # codebook = torch.round(model.vector_quantization.embedding.weight.data, decimals=3)
        complete_q_tensor = torch.round(torch.tensor(np.load("created_files/patch_q_emb_v1.npy"), dtype=torch.float32), decimals=3)
        complete_q_indices = np.load("created_files/patch_q_ind_v1.npy")
    elif mode == "VQ-VAE-Yannik-Patch":
        mlp_path = "MLPs/my_trained_mlp_on_y_patch.ckpt"
        vqvae_path = "Y-VQ-VAE-Patch-best.ckpt"
        my_trained_mlp = MLP(input_size=16, output_size=2, in_dim=32, hidden_sizes=512)
        my_trained_mlp.load_state_dict(torch.load(mlp_path))
        my_trained_mlp.eval()
        model, hparams = get_model_patch("VQ-VAE-Patch", vqvae_path)
        model.eval()
        codebook = torch.round(model.vector_quantization.embedding.weight.data, decimals=3)
        complete_q_tensor = torch.round(torch.tensor(np.load("created_files/y_patch_q_emb.npy"), dtype=torch.float32), decimals=3)
        complete_q_indices = np.load("created_files/y_patch_q_ind.npy")
    return my_trained_mlp, model, codebook, complete_q_tensor, complete_q_indices, hparams
    
def get_dataloaders_and_datasets():
    data_dict = get_val_test_ids()
    val_ids = data_dict["val_ids"]
    test_ids = data_dict["test_ids"]
    val_ids = [DataSplitId(experiment=item[0], welding_run=item[1])
                for item in val_ids]
    test_ids = [DataSplitId(experiment=item[0], welding_run=item[1])
                for item in test_ids]
    data_module = ASIMoWDataModule(task="classification", batch_size=128, n_cycles=1, val_data_ids=val_ids, test_data_ids=test_ids)

    data_module.setup('fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    train_data = train_loader.dataset.data
    val_data = val_loader.dataset.data
    test_data = test_loader.dataset.data

    train_labels = train_loader.dataset.labels
    val_labels = val_loader.dataset.labels
    test_labels = test_loader.dataset.labels
    return train_loader, val_loader, test_loader, train_data, val_data, test_data, train_labels, val_labels, test_labels

def get_embedding_files():
    # load embeddings from file
    q_embeddings = np.load("created_files/train_loader_q_embeddings.npy")
    q_indices = np.load("created_files/train_loader_q_indices.npy")

    # load all_saliency_maps and all_saliency_maps_bad from npy
    sm = np.load("created_files/all_saliency_maps.npy")
    sm_bad = np.load("created_files/all_saliency_maps_bad.npy")
    sm_embed_mean = np.load("created_files/all_saliency_maps_embed_mean.npy")
    sm_embed_mean_bad = np.load("created_files/all_saliency_maps_embed_mean_bad.npy")
    sm_dim_mean = np.load("created_files/all_saliency_maps_dim_mean.npy")
    sm_dim_mean_bad = np.load("created_files/all_saliency_maps_dim_mean_bad.npy")

    q_tensor = torch.round(torch.tensor(q_embeddings, dtype=torch.float32), decimals=3)
    return q_tensor, q_indices, sm, sm_bad, sm_embed_mean, sm_embed_mean_bad, sm_dim_mean, sm_dim_mean_bad

def get_patch_embedding_files():
    # load embeddings from file
    q_embeddings = np.load("created_files/patch_train_loader_q_embeddings.npy")
    q_indices = np.load("created_files/patch_train_loader_q_indices.npy")
    q_tensor = torch.round(torch.tensor(q_embeddings, dtype=torch.float32), decimals=3)
    return q_tensor, q_indices  

def send_through_model(model, input_data, reconstruct=True):
    with torch.no_grad():
        permuted_original = input_data.permute(0, 2, 1)
        z_e = model.encoder(permuted_original)
        encoded_data = z_e.permute(0,2,1)
        embedding_loss, quantized_data, perplexity, _, q_indices = model.vector_quantization(encoded_data)
        z_q = quantized_data.permute(0,2,1)
        if model.decoder_type == "Linear":
            z_q = z_q.reshape(z_q.shape[0], z_q.shape[1] * z_q.shape[2])
        x_hat = None
        recon_error = None
        loss = None
        if reconstruct:
            x_hat = model.decoder(z_q)
            if model.decoder_type == "Conv":
                x_hat = x_hat.permute(0, 2, 1)
            recon_error = F.mse_loss(x_hat, input_data)
            loss = recon_error + embedding_loss

    return embedding_loss, x_hat, perplexity, quantized_data, q_indices, recon_error, loss

def send_through_patch_model(model, input_data, reconstruct=False):
    with torch.no_grad():
        x = model.patch_embed(input_data)
        z_e = model.encoder(x)
        embedding_loss, z_q, perplexity, _, q_indices = model.vector_quantization(z_e)
        x_hat = None
        recon_error = None
        loss = None
        if reconstruct:
            x_hat = model.decoder(z_q.permute(0, 2, 1))
            x_hat = model.reverse_patch_embed(x_hat)
            recon_error = F.mse_loss(x_hat, input_data)
            loss = recon_error + embedding_loss
    return embedding_loss, x_hat, perplexity, z_q, q_indices, recon_error, loss


def send_through_decoder(q_data, model, original_data=None, label = "VQ-VAE"):
    with torch.no_grad():
        if label != "VQ-VAE":
            x_hat = model.decoder(q_data.permute(0, 2, 1))
            reconstructed_original = model.reverse_patch_embed(x_hat)
            recon_error = F.mse_loss(reconstructed_original, original_data.unsqueeze(0))
            # loss = recon_error + embedding_loss
        else:
            # z_q = q_data.reshape(1, q_data.shape[0], q_data.shape[1])
            z_q = q_data.permute(0,2,1)

            if model.decoder_type == "Linear":
                z_q = z_q.reshape(z_q.shape[0], z_q.shape[1] * z_q.shape[2])
            x_hat = model.decoder(z_q)
            if model.decoder_type == "Conv":
                x_hat = x_hat.permute(0, 2, 1)
            reconstructed_original = x_hat.reshape(-1, model.seq_len, model.input_dim)
            # reconstructed_original = reconstructed_original.reshape(200,2)
            recon_error = None
            if original_data is not None: 
                recon_error = F.mse_loss(reconstructed_original, original_data.unsqueeze(0))
    return reconstructed_original, recon_error

def generate_saliency_map(model, input_data, target_class):
    input_data.requires_grad = True
    print(input_data.unsqueeze(0).shape)
    logits = model(input_data.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.tensor([target_class]))
    loss.backward()
    saliency_map = input_data.grad.squeeze().abs()
    return saliency_map

def plot_saliency_map(all_saliency_maps, model, index=0):
    sm = torch.tensor(all_saliency_maps[index])
    sm_summed_or_mean = sm.mean(dim=1, keepdim=True)
    sm_dim_summed_or_mean = sm.mean(dim=0, keepdim=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(sm, cmap='Wistia')
    plt.title(f'Saliency Map mit  {model.enc_out_len} x {model.embedding_dim}')

    plt.subplot(1, 3, 2)
    plt.xticks([])
    plt.imshow(sm_summed_or_mean, cmap='Wistia')
    plt.title(f'Mean über {model.enc_out_len} Embeddings')

    plt.subplot(1, 3, 3)
    plt.yticks([])
    plt.imshow(sm_dim_summed_or_mean, cmap='Wistia')
    plt.title(f'Mean über {model.embedding_dim} Dimensionen')
    
    plt.show()

def plot_reconstruction_difference(reconstructed_original, reconstructed_changed, original, label, plot_original=False):
    difference = torch.sum(torch.abs(reconstructed_original - reconstructed_changed), dim=1)
    colormap = plt.colormaps['Wistia']
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 7)
    ax2 = ax.twinx()
    bars = ax.bar(np.arange(len(reconstructed_original)), [1] * len(reconstructed_original), color=colormap(difference), alpha=0.7)
    ax.set_ylim(0, 1)
    ax2.set_ylim(-5, 5)
    ax.set_xlabel('Index')
    ax.set_ylabel('Difference')
    ax2.plot(reconstructed_original)
    ax2.plot(reconstructed_changed)
    # label fig
    ax.set_title(f'Difference between Original and Changed Reconstruction for {label}')
    if plot_original: ax2.plot(original)
    plt.show()
    return difference

def alter_q_data(q_data, original_data, alter_range, alter_embedding, model, model_hparams, label, plot_original=False):
    if 'use_improved_vq' not in model_hparams: model_hparams['use_improved_vq'] = None
    if model_hparams["use_improved_vq"]:
        codebook = torch.round(model.vector_quantization.vq.codebooks[0], decimals=3)
    else:
        codebook = torch.round(model.vector_quantization.embedding.weight.data, decimals=3)
    changed_q_data = q_data.clone().detach()
    # codebook = torch.round(model.vector_quantization.embedding.weight.data, decimals=3)
    # replacing four embeddings with least common Codebook Entry number 165
    for i in alter_range:
        changed_q_data[i] = codebook[alter_embedding] 
    reconstructed_original, _= send_through_decoder(model=model, q_data=q_data.unsqueeze(0), original_data = original_data, label=label)
    reconstructed_changed, _= send_through_decoder(model=model, q_data=changed_q_data.unsqueeze(0), original_data = original_data, label=label)
    difference = plot_reconstruction_difference(reconstructed_original=reconstructed_original[0], reconstructed_changed=reconstructed_changed[0], original=original_data, label=label, plot_original=plot_original)