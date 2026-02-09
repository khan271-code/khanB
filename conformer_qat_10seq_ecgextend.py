import os
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import torch.ao.quantization as quantization
import numpy as np
import pandas as pd

from accelerate import Accelerator
from collections import OrderedDict
from torch_geometric.data import Dataset
import threading
import os

import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Split data into train, test, and validation datasets
def split_data(data_list, split=0.6, seed=0):
    """
    Split dataset into train, validation, and test dataset.
    """
    generator1 = torch.Generator().manual_seed(seed)
    vsplit = (1 - split) / 2
    tsplit = 1 - split - vsplit
    train_set, val_set, test_set = torch.utils.data.random_split(data_list, [split, vsplit, tsplit], generator1)
    return train_set, val_set, test_set

class RigidTransform(nn.Module):
    """Differentiable rigid transformation module"""
    def __init__(self, epiapex_idx=37, basering_start=25, basering_end=37):
        super().__init__()
        self.register_buffer("epiapex_idx", torch.tensor(epiapex_idx))
        self.register_buffer("basering_indices",
                             torch.arange(basering_start, basering_end))

    def forward(self, coordinates):
        B, T, N, _ = coordinates.shape
        apex = coordinates[:, 0, self.epiapex_idx, :]
        base = torch.mean(coordinates[:, 0, self.basering_indices, :], dim=1)

        x_axis = F.normalize(base - apex, dim=-1)
        y_ref = coordinates[:, 0, 25, :] - apex
        z_axis = F.normalize(torch.linalg.cross(x_axis, y_ref), dim=-1)
        y_axis = F.normalize(torch.linalg.cross(z_axis, x_axis), dim=-1)

        rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)
        translated = coordinates - apex.view(B, 1, 1, 3)
        rotated = torch.einsum('btnj,bjk->btnk', translated, rot_mat)
        return rotated

class ShardedDataset(Dataset):
    # --- MODIFIED: Simplified for direct prediction ---
    def __init__(self, root, shards, prefix="ecglvmesh_10sec_", items_per_shard=100, cache_size=500):
        super().__init__(root)
        self.root = root
        self.shards = shards
        self.prefix = prefix
        self.items_per_shard = items_per_shard
        self.total_items = len(shards) * self.items_per_shard
        self.cache_size = cache_size
        self.shard_cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.rigid_transform = RigidTransform()

    def load_shard(self, shard_idx):
        path = os.path.join(self.root, f'{self.prefix}{shard_idx + 1}.pt')
        return torch.load(path, map_location='cpu',weights_only=False)

    def preload(self):
        ctr = 0
        print(f"Preloading data {len(self.shards)} into cache of {self.cache_size}",flush=True)
        for shrd in self.shards:
            self.get_shard(shrd)
            ctr +=1
            if ctr > self.cache_size:
                break
        print(f"Preload of {len(self.shards)} into cache of {self.cache_size} completed. Num shards {ctr}",flush=True)

    def get_shard(self, shard_idx):
        with self.cache_lock:
            if shard_idx in self.shard_cache:
                self.shard_cache.move_to_end(shard_idx)
                return self.shard_cache[shard_idx]
            else:
                shard_data = self.load_shard(shard_idx)
                self.shard_cache[shard_idx] = shard_data
                if len(self.shard_cache) > self.cache_size:
                    self.shard_cache.popitem(last=False)
                return shard_data

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        shard_idx = idx // self.items_per_shard
        item_idx = idx % self.items_per_shard
        shard_data = self.get_shard(shard_idx)
        data_item = shard_data[item_idx]

        for key, item in data_item:
            if isinstance(item, torch.Tensor):
                setattr(data_item, key, item.detach().float())

        # --- MODIFIED: The ground truth is now the final, aligned mesh ---
        target_reshaped = data_item.y.view(1, 50, 98, 3)
        aligned_target = self.rigid_transform(target_reshaped)
        
        data_item.y = aligned_target.squeeze(0)

        return data_item

def create_adjacency_info():
    adj_pairs = []
    def add_pair(a, b):
        if (a, b) not in adj_pairs and (b, a) not in adj_pairs:
            adj_pairs.append((a, b))

    endo_slices = {'apex': [0], 'third': list(range(1, 13)), 'fourth': list(range(13, 25)), 'fifth': list(range(25, 37)), 'sixth': list(range(74, 86))}
    epi_slices = {'apex': [37], 'third': list(range(38, 50)), 'fourth': list(range(50, 62)), 'fifth': list(range(62, 74)), 'sixth': list(range(86, 98))}
    
    apex_endo, apex_epi = endo_slices['apex'][0], epi_slices['apex'][0]
    for node in endo_slices['third']: add_pair(apex_endo, node)
    add_pair(apex_endo, apex_epi)
    for node in epi_slices['third']: add_pair(apex_epi, node)

    def connect_slice(s1, s2, es1=None, es2=None):
        n = len(s1)
        for i in range(n):
            add_pair(s1[i], s1[(i + 1) % n]); add_pair(s1[i], s2[i])
            if es1: add_pair(s1[i], es1[i]); add_pair(es1[i], es1[(i + 1) % n])
            if es2: add_pair(s2[i], es2[i]); add_pair(es2[i], es2[(i + 1) % n])
            if es1 and es2: add_pair(es1[i], es2[i])
    
    connect_slice(endo_slices['third'], endo_slices['fourth'], epi_slices['third'], epi_slices['fourth'])
    connect_slice(endo_slices['fourth'], endo_slices['fifth'], epi_slices['fourth'], epi_slices['fifth'])
    connect_slice(endo_slices['fifth'], endo_slices['sixth'], epi_slices['fifth'], epi_slices['sixth'])

    N = 98
    indices, values = [], []
    for point in range(N):
        neighbors = [j for (i, j) in adj_pairs if i == point] + [i for (i, j) in adj_pairs if j == point]
        k = len(neighbors)
        indices.append([point, point]); values.append(-1.0)
        for n in neighbors: indices.append([point, n]); values.append(1.0 / k if k > 0 else 0)
    
    indices = torch.LongTensor(indices).t()
    values = torch.FloatTensor(values)
    L = torch.sparse_coo_tensor(indices, values, (N, N)).to_dense()
    return adj_pairs, L

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class DepthwiseConv1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw_conv = nn.Conv1d(dim, dim, kernel_size=1)
    def forward(self, x):
        x = x.transpose(1, 2); x = self.pw_conv(self.dw_conv(x)); return x.transpose(1, 2)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "embed_dim must be divisible by num_heads"
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x_norm = self.norm(x)
        B, S, D = x_norm.shape
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q, k, v = [t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        x_attn = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.proj(x_attn)

class ConformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.ff1 = FeedForward(self.embed_dim, config['dropout'])
        self.conv = DepthwiseConv1d(self.embed_dim)
        self.attn = MultiHeadAttention(self.embed_dim, config['num_heads'])
        self.ff2 = FeedForward(self.embed_dim, config['dropout'])
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)
        self.norm4 = nn.LayerNorm(self.embed_dim)
        self.film_proj = nn.Linear(config['embed_dim'], self.embed_dim * 2)

    def forward(self, x, metadata_emb):
        gamma, beta = self.film_proj(metadata_emb).chunk(2, dim=-1)
        gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        x = gamma * self.norm1(x) + beta
        x = x + self.ff1(self.norm2(x))
        x = x + self.conv(self.norm3(x))
        x = x + self.attn(x)
        x = x + self.ff2(self.norm4(x))
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, factor=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_c = in_channels if i == 0 else out_channels
            layers.append(nn.Conv1d(in_c, out_channels, kernel_size=factor*2+1, stride=factor, padding=factor))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class EnhancedECG2Mesh(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config['embed_dim']
        self.time_points = config['time_points']

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        
        self.metadata_input_dim = 1 + 1 + 1 + 12 # age, gender, gfp, 12-lead amps
        self.metadata_encoder = nn.Sequential(
            nn.Linear(self.metadata_input_dim, config['demog_dim']), nn.ReLU(),
            nn.Linear(config['demog_dim'], config['demog_dim']), nn.ReLU(),
            nn.Linear(config['demog_dim'], self.embed_dim)
        )
        
        self.input_proj = nn.Linear(config['ecg_embedding_dim'], self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim, config['ecg_embedding_sequence_len'])
        
        self.downsampler = nn.Sequential(
            DownsampleBlock(self.embed_dim, self.embed_dim, num_layers=1, factor=4),
            DownsampleBlock(self.embed_dim, self.embed_dim, num_layers=1, factor=4),
            DownsampleBlock(self.embed_dim, self.embed_dim, num_layers=1, factor=4),
            nn.AdaptiveAvgPool1d(self.time_points)
        )
        
        self.blocks = nn.ModuleList([ConformerBlock(config) for _ in range(config['num_layers'])])
        
        # --- MODIFIED: Single head for direct regression. No Tanh for unbounded output ---
        self.shape_head = nn.Sequential(
            nn.Linear(self.embed_dim, config['hidden_dim']), nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['hidden_dim']), nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['num_points'] * 3)
        )
        
    def forward(self, ecg_embedding, age, gender, peak_gfp, peak_amplitudes):
        x = self.quant(ecg_embedding)
        
        batch_size = age.shape[0]
        age_r = age.view(batch_size, -1)
        gender_r = gender.view(batch_size, -1)
        peak_gfp_r = peak_gfp.view(batch_size, -1)
        peak_amplitudes_r = peak_amplitudes.view(batch_size, -1)
        metadata = torch.cat([age_r, gender_r, peak_gfp_r, peak_amplitudes_r], dim=1)

        metadata_emb = self.metadata_encoder(metadata)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.downsampler(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x, metadata_emb)
        
        output_mesh = self.shape_head(x).view(-1, self.time_points, self.config['num_points'], 3)
        
        output_mesh = self.dequant(output_mesh)

        return output_mesh

def train_model(config_name, config, train_loader, val_loader, laplacian_matrix, accelerator, checkpoint=True, event=None):
    device = accelerator.device
    model = EnhancedECG2Mesh(config).to(device)
    
    if config.get('qat', False):
         print("Preparing model for Quantization Aware Training...")
         model.train()
         model.qconfig = quantization.get_default_qat_qconfig('qnnpack')
         quantization.fuse_modules(model, [
            ['metadata_encoder.0', 'metadata_encoder.1'], 
            ['metadata_encoder.2', 'metadata_encoder.3'],
            ['shape_head.0', 'shape_head.1'],
            ], inplace=True)
         quantization.prepare_qat(model, inplace=True)
    else:
        if accelerator.is_local_main_process:
            model = torch.compile(model, mode="reduce-overhead")

    model = model.float()
        
    checkpointdir = '/eresearch/ecg-echo-xai/rjag008/ukbb/checkpoints'
    if not os.path.exists(checkpointdir):
       raise FileNotFoundError(f"{checkpointdir} does not exist!!")

    if os.path.exists(f"{checkpointdir}/checkpoint_{config_name}.pt"):
        try:
            state_dict, _ = torch.load(f"{checkpointdir}/checkpoint_{config_name}.pt", map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            print("Model loaded from checkpoint")
        except Exception as e: print(f"Could not load checkpoint: {e}")

    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config['weight_decay'])
    
    warmup_epochs = 5
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"] - warmup_epochs, eta_min=1e-7)

    
    shape_mse_loss_func = nn.MSELoss()
    
    L = laplacian_matrix.to(device)
    
    non_apex_indices = [i for i in range(config['num_points']) if i not in [0, 37]]
    laplacian_mask = torch.zeros(config['num_points'], 1, device=device)
    laplacian_mask[non_apex_indices] = 1.0


    train_loader, val_loader, model, optimizer, warmup_scheduler, main_scheduler = accelerator.prepare(
        train_loader, val_loader, model, optimizer, warmup_scheduler, main_scheduler
    )
    if event: event.set()

    best_val_mse = 1e-3#float("inf")
    patience, epochs_without_improvement = 250, 0
    timeseq_indices = torch.linspace(0, 49, steps=config['time_points']).long()
    laplacian_loss_weight = config['laplacian_loss_weight']
    
    original_time_steps = 50

    for epoch in range(config["epochs"]):
        model.train()
        epoch_train_loss, epoch_shape_mse_loss, epoch_laplacian_loss = 0, 0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}", unit="batch", disable=not accelerator.is_local_main_process) as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                
                batch_size = batch.age.shape[0]
                # --- MODIFIED: The target is now the final aligned mesh ---
                target_sliced = batch.y.view(batch_size, original_time_steps, config['num_points'], 3)[:, timeseq_indices, :, :]
                B, T, N, C = target_sliced.shape
                
                pred_mesh = model(
                    batch.embedding, batch.age, batch.gender, 
                    batch.peak_gfp, batch.peak_amplitudes
                )
                
                shape_mse = shape_mse_loss_func(pred_mesh, target_sliced)
                
                pred_mesh_flat = pred_mesh.view(B * T, N, C)
                laplacian_values = torch.norm(torch.matmul(L, pred_mesh_flat), dim=-1)
                
                masked_laplacian_values = laplacian_values * laplacian_mask.squeeze(-1)
                laplacian_loss = torch.mean(masked_laplacian_values)

                total_loss = shape_mse + laplacian_loss_weight * laplacian_loss
                
                accelerator.backward(total_loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_train_loss += total_loss.item()
                epoch_shape_mse_loss += shape_mse.item()
                epoch_laplacian_loss += laplacian_loss.item()
                tepoch.set_postfix({'ShapeMSE': shape_mse.item(), 'Total': total_loss.item()})
        
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        model.eval()
        val_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_size = batch.age.shape[0]
                target_sliced = batch.y.view(batch_size, original_time_steps, config['num_points'], 3)[:, timeseq_indices, :, :]

                pred_mesh = model(
                    batch.embedding, batch.age, batch.gender, 
                    batch.peak_gfp, batch.peak_amplitudes
                )
                
                val_mse += shape_mse_loss_func(pred_mesh, target_sliced).item()

        avg_val_mse = val_mse / len(val_loader)

        if accelerator.is_local_main_process:
            print(f"\nConfig {config_name}, Epoch {epoch + 1}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Train - ShapeMSE: {epoch_shape_mse_loss/len(train_loader):.6f}, Total: {epoch_train_loss/len(train_loader):.6f}")
            print(f"Val   - True MSE: {avg_val_mse:.6f}\n", flush=True)

            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                epochs_without_improvement = 0
                if checkpoint:
                    print(f"New best validation MSE: {best_val_mse:.6f}. Saving checkpoint.")
                    if config.get('qat', False):
                        model.eval()
                        quantized_model = quantization.convert(model.to('cpu'))
                        accelerator.save([quantized_model.state_dict(), config], f"{checkpointdir}/quantized_model_{config_name}.pt")
                        model.to(device)
                    else:
                         accelerator.save([accelerator.unwrap_model(model).state_dict(), config], f"{checkpointdir}/checkpoint_{config_name}.pt")

            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered on {config_name}.")
                    break

if __name__ =='__main__':
    config = {
        'embed_dim': 512, 'hidden_dim': 1024, 'num_heads': 8, 'num_layers': 8,
        'num_points': 98, 
        'ecg_embedding_dim': 768,
        'ecg_embedding_sequence_len': 312, 
        'time_points': 10, 'demog_dim': 128, 'dropout': 0.2, 'learning_rate': 1e-4, 
        'epochs': 2500, 'batch_size': 1024*2, 'laplacian_loss_weight': 0.2, # Increased weight for regularization
        'weight_decay': 1e-5, 'qat': True 
    }
    
    _, laplacian_matrix = create_adjacency_info()

    data_list = range(1,191)
    train_data_indices, val_data_indices, _ = split_data(data_list, split=0.8)
    
    accelerator = Accelerator(mixed_precision="no" if config['qat'] else "fp16")
    print(f"Starting training on {accelerator.device}\n{config}")

    shrad_root = '/eresearch/ecg-echo-xai/rjag008/ukbbdata/shrad_10sec/'

    train_dataset = ShardedDataset(shrad_root, list(train_data_indices), prefix="ecglvmesh_10sec_")
    train_dataset.preload()
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = ShardedDataset(shrad_root, list(val_data_indices), prefix="ecglvmesh_10sec_", cache_size=100)
    val_dataset.preload()
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=4, pin_memory=True)

    train_model("QAT_Direct", config, train_loader, val_loader, laplacian_matrix, accelerator)
