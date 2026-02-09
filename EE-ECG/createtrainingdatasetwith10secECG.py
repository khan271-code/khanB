import torch
import torch.nn as nn
from torch_geometric.data import Data
import json
torch.set_printoptions(precision=8)
import os, pickle
import xmltodict
import numpy as np
import torch
import neurokit2 as nk
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from fairseq_signals.models import build_model_from_checkpoint

landmarks_dir = '/eresearch/ecg-echo-xai/rjag008/ukbb/Landmarks'
ecgdata_dir   = '/eresearch/ecg-echo-xai/rjag008/ukbb/ECGdata/data'

ecgfm = build_model_from_checkpoint('/eresearch/ecg-echo-xai/rjag008/ukbb/ckpts/mimic_iv_ecg_physionet_pretrained.pt')
ecgfm.to(torch.float)
targetcs_uv = torch.tensor([(0, 1, 0), (-1, 0, 0), (0, 0, 1)],dtype=float)


# Remove thickness outliers
def remove_outliers_batched(thicknesses, threshold=1.5):
    q1 = thicknesses.quantile(0.25, dim=1, keepdim=True)
    q3 = thicknesses.quantile(0.75, dim=1, keepdim=True)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    mask = (thicknesses >= lower) & (thicknesses <= upper)
    cleaned = torch.where(mask, thicknesses, torch.tensor(float('nan'), device=thicknesses.device))
    avg = torch.nanmean(cleaned, dim=1)
    return avg

def fit_paraboloid_coeffs(points):
    """
    Fit a paraboloid z = ax² + by² + cxy + dx + ey + f to (B, N, 3) points.
    Return coefficients of shape (B, 6)
    """
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    ones = torch.ones_like(x)
    A = torch.stack([x**2, y**2, x * y, x, y, ones], dim=-1)  # (B, N, 6)
    z = z.unsqueeze(-1)
    AtA = torch.matmul(A.transpose(1, 2), A)
    Atz = torch.matmul(A.transpose(1, 2), z)
    coeffs = torch.linalg.solve(AtA, Atz).squeeze(-1)  # (B, 6)

    # Flip if opening upward
    a, b = coeffs[:, 0], coeffs[:, 1]
    opens_up = (a > 0) & (b > 0)
    coeffs[opens_up] *= -1
    return coeffs

def paraboloid_z(xyz, coeffs):
    """
    Evaluate z = ax² + by² + cxy + dx + ey + f over batched input points
    Args:
        xyz: Tensor of shape (B, N, 3) or (B, 3)
        coeffs: Tensor of shape (B, 6)
    Returns:
        z: Tensor of shape (B, N) or (B,)
    """
    if xyz.dim() == 2:
        x, y = xyz[:, 0], xyz[:, 1]
        a, b, c, d, e, f = coeffs.unbind(-1)
    elif xyz.dim() == 3:
        x, y = xyz[..., 0], xyz[..., 1]
        coeffs = coeffs.unsqueeze(1)  # (B, 1, 6) for broadcasting
        a, b, c, d, e, f = coeffs.unbind(-1)  # Each (B, N)
    else:
        raise ValueError("xyz must be of shape (B, 3) or (B, N, 3)")

    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

def update_apex_along_axis(points):
    """
    Update endo and epi apex such that:
    - They lie on the fitted surface
    - They are separated along the anatomical axis
    - Distance equals average wall thickness near apex
    """
    endo_surf = points[:, 1:37]
    epi_surf = points[:, 38:74]
    endo_coeffs = fit_paraboloid_coeffs(endo_surf)
    epi_coeffs  = fit_paraboloid_coeffs(epi_surf)
    endo_base = points[:, 25:37].mean(dim=1)
    epi_base  = points[:, 61:73].mean(dim=1)
    axis = F.normalize(epi_base - endo_base, dim=-1)
    endo_ring = points[:, 1:7]
    epi_ring  = points[:, 38:44]
    thickness = (epi_ring - endo_ring).norm(dim=-1)
    avg_thickness = remove_outliers_batched(thickness)
    endo_guess = points[:, 0]
    mid_guess = endo_guess + 0.5 * avg_thickness.unsqueeze(-1) * axis
    t_vals = torch.linspace(-1.5, 1.5, steps=100, device=points.device).view(1, -1, 1)
    axis_exp = axis.unsqueeze(1)
    base_pts = mid_guess.unsqueeze(1) + t_vals * axis_exp
    endo_xy = base_pts - axis_exp * (avg_thickness / 2).unsqueeze(1).unsqueeze(-1)
    epi_xy  = base_pts + axis_exp * (avg_thickness / 2).unsqueeze(1).unsqueeze(-1)
    endo_z = paraboloid_z(endo_xy, endo_coeffs).unsqueeze(-1)
    epi_z  = paraboloid_z(epi_xy, epi_coeffs).unsqueeze(-1)
    endo_pts = torch.cat([endo_xy[..., :2], endo_z], dim=-1)
    epi_pts  = torch.cat([epi_xy[..., :2], epi_z], dim=-1)
    actual_dist = (epi_pts - endo_pts).norm(dim=-1)
    error = (actual_dist - avg_thickness.unsqueeze(1)).abs()
    best_idx = error.argmin(dim=1)
    idx = best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3)
    endo_apex = endo_pts.gather(1, idx).squeeze(1)
    epi_apex  = epi_pts.gather(1, idx).squeeze(1)
    updated = points.clone()
    updated[:, 0]  = endo_apex
    updated[:, 37] = epi_apex
    return updated


def translate_rotate_points_3d(unit_vectors_from, unit_vectors_to, points):
    rotation_matrix = torch.linalg.inv(unit_vectors_from) @ unit_vectors_to
    transformed_points = torch.matmul(points, rotation_matrix)
    return transformed_points

def rigid_transform_coordinates(coordinates, targetcs_uv):
    epiapexix = 37
    baseringix = torch.arange(25, 37, dtype=torch.long)
    yaxisix = 25
    apexp = coordinates[0, epiapexix, :]
    basep = torch.mean(coordinates[0, baseringix, :], dim=0)
    yaxisp = coordinates[0, yaxisix, :]
    xaxis = basep - apexp
    yaxis = yaxisp - basep
    xaxis = xaxis / torch.norm(xaxis)
    yaxis = yaxis / torch.norm(yaxis)
    zaxis = torch.linalg.cross(xaxis, yaxis)
    zaxis = zaxis / torch.norm(zaxis)
    unit_vectors_from = torch.stack([xaxis, yaxis, zaxis])
    off_coord = coordinates - apexp
    newcoord = translate_rotate_points_3d(unit_vectors_from, targetcs_uv, off_coord)
    return newcoord
    
def calculate_ecg_features(ecg_tensor):
    ecg_np = ecg_tensor.numpy()
    gfp = np.std(ecg_np, axis=0)
    peak_gfp = np.max(gfp)
    peak_amplitudes = np.ptp(ecg_np, axis=1)
    features = {
        'peak_gfp': torch.tensor([peak_gfp]).float(),
        'peak_amplitudes': torch.from_numpy(peak_amplitudes).float()
    }
    return features

def extract_ecg_data(filename):
    """
    Extracts the full 10-second, 12-lead ECG strip and metadata from the XML file.
    """
    with open(filename,'rb') as xml:
        ECG = xmltodict.parse(xml.read().decode('utf8'))
        od = ECG['CardiologyXML']
        age = od['PatientInfo']['Age']['#text']
        gender = 1.0 if od['PatientInfo']['Gender'].strip().upper()=='MALE' else 0.0        
        ecgm = od['StripData']['WaveformData']    
        ecg_data = {}
        for e in ecgm:
            ecg_data[e['@lead']] = list(map(float,e['#text'].replace("\n",'').replace("\t",'').split(",")))
        
        samplingrate = 500
        try:
            samplingrate = int(od['StripData']['SampleRate']['#text'].strip())
        except:
            pass

        lead_order = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
        ecg_tensor = torch.tensor([ecg_data[l] for l in lead_order], dtype=torch.float)
        
        # Ensure the signal is 10 seconds (5000 samples at 500Hz)
        num_samples = ecg_tensor.shape[1]
        target_samples = 10 * samplingrate
        if num_samples > target_samples:
            ecg_tensor = ecg_tensor[:, :target_samples]
        elif num_samples < target_samples:
            # Pad with zeros if shorter
            padding = torch.zeros(ecg_tensor.shape[0], target_samples - num_samples)
            ecg_tensor = torch.cat([ecg_tensor, padding], dim=1)

        cleaned_leads = []
        for i in range(ecg_tensor.shape[0]):
            cleaned_lead = nk.ecg_clean(ecg_tensor[i, :].numpy(), sampling_rate=samplingrate)
            # --- FIXED: Use .copy() to resolve negative stride issue ---
            cleaned_leads.append(torch.from_numpy(cleaned_lead.copy()).float())
        ecg_tensor_cleaned = torch.stack(cleaned_leads, dim=0)

        return ecg_tensor_cleaned, int(age), gender

def extractContours(contourfile):
    raw_points = torch.from_numpy(np.load(contourfile)['points'])
    apex_updated_contours = update_apex_along_axis(raw_points)
    landmark_coordinates = rigid_transform_coordinates(apex_updated_contours,targetcs_uv)
    return landmark_coordinates.reshape((-1,294))

with open('data/ecgfile2niimap.pkl','rb') as ecg:
    ecgfiles = pickle.load(ecg)

print(f"Number of files {len(ecgfiles)}")    
trainingdata = []
failedfiles = []
shrad_size = 100
shrad_ctr = 1
total = 0

filekeys = list(ecgfiles.keys())
pbar = tqdm(filekeys, desc="Processing files ")
for k in pbar:
    v = ecgfiles[k]
    try:
        response = extractContours(os.path.join(landmarks_dir,v))
        
        ecg_tensor, age, gender = extract_ecg_data(os.path.join(ecgdata_dir,k))
        
        ecg_features = calculate_ecg_features(ecg_tensor)
        
        padding_mask = torch.zeros(1, ecg_tensor.shape[1], dtype=torch.bool)
        
        embeddingfm = ecgfm.extract_features(ecg_tensor.unsqueeze(0), padding_mask)['x']
        
        trainingdata.append(Data(
            x=ecg_tensor.unsqueeze(0).half(), 
            y=response.unsqueeze(0).half(), 
            embedding=embeddingfm.half(),
            age=torch.tensor([age]).half(),
            gender=torch.tensor([gender]).half(),
            peak_gfp=ecg_features['peak_gfp'].half(),
            peak_amplitudes=ecg_features['peak_amplitudes'].half()
        ))
            
        if len(trainingdata) >= shrad_size:
            pbar.set_postfix({"total": total})
            torch.save(trainingdata, f'data/shrad_10sec/ecglvmesh_10sec_{shrad_ctr}.pt')   
            shrad_ctr += 1
            total += len(trainingdata)
            trainingdata = []

    except Exception as ex:
        print(f"* Failed on {os.path.join(ecgdata_dir,k)} with contour {v}. Error: {ex}")
        import traceback
        traceback.print_exc()
        failedfiles.append([k,v])

if trainingdata:
    total += len(trainingdata)
    torch.save(trainingdata, f'data/shrad_10sec/ecglvmesh_10sec_{shrad_ctr}.pt')   

torch.save(failedfiles, f'data/failedfiles_10sec.pt')   
print(f"Generated {total} files and placed in {shrad_ctr} shards")
