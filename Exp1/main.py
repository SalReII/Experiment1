import sys
import os
import torch
import pandas as pd
import numpy as np
import cv2
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt

model_path = "/Users/salrei/Documents/Diss/Exp1/model/ser-fiq-pytorch"
sys.path.append(os.path.abspath(model_path))
from backbones.iresnet import iresnet50


def get_model(checkpoint_path, device):
    model = iresnet50(dropout=0.4, num_features=512, use_se=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_serfiq(img_tensor, model, device, T=30):
    model.eval() 

    def enable_dropout(m):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.apply(enable_dropout)

    embeddings = []

    with torch.no_grad():
        for _ in range(T):
            feat = model(img_tensor)
            feat = torch.nn.functional.normalize(feat, dim=1)
            embeddings.append(feat.cpu().numpy())

    embeddings = np.vstack(embeddings)

    sim_matrix = np.dot(embeddings, embeddings.T)

    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    sims = sim_matrix[mask]

    score = np.mean(sims)

    return float(score)

def run_verification(clean_features, degraded_features, far_target=1e-3):
    pos_sims, neg_sims = [], []

    ids = list(degraded_features.keys())
    clean_names = list(clean_features.keys())

    for name in ids:
        if name in clean_features:
            f1 = clean_features[name].flatten()
            f2 = degraded_features[name].flatten()
            sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            pos_sims.append(sim)

    for i in range(len(ids)):
        f_deg = degraded_features[ids[i]].flatten()
        for j in range(len(clean_names)):
            if ids[i] == clean_names[j]:
                continue
            f_cln = clean_features[clean_names[j]].flatten()
            sim = np.dot(f_cln, f_deg) / (np.linalg.norm(f_cln) * np.linalg.norm(f_deg))
            neg_sims.append(sim)

    all_sims = np.array(pos_sims + neg_sims)
    all_labels = np.array([1]*len(pos_sims) + [0]*len(neg_sims))

    fpr, tpr, _ = roc_curve(all_labels, all_sims)
    tar = np.interp(far_target, fpr, tpr)

    return float(tar), auc(fpr, tpr)


def compute_all_metrics(img_bgr, model, device):
    img_112 = cv2.resize(img_bgr, (112, 112))
    gray = cv2.cvtColor(img_112, cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    q_sharp = np.tanh(lap_var / 150.0)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    noise_est = float((gray.astype(np.float32) - blurred).std())
    q_noise = np.exp(-noise_est / 15.0)
    q_cont = np.tanh(gray.std() / 40.0)
    h, w = img_bgr.shape[:2]
    q_res = np.clip(min(h, w) / 112.0, 0, 1)

    img_t = torch.from_numpy(img_112).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        feat = model(img_t)

    feat_np = feat.cpu().numpy()
    norm = float(torch.norm(feat))

    serfiq_score = compute_serfiq(img_t, model, device, T=10)
    q_embed = np.tanh(norm / 20.0)

    q_final = (
        0.35 * q_embed +
        0.25 * q_sharp +
        0.20 * q_noise +
        0.10 * q_res +
        0.10 * q_cont
    )

    return {
        'q_nn': float(q_final),
        'serfiq': serfiq_score,
        'norm': norm,
        'lap': lap_var,
        'feat': feat_np
    }

def make_all_plots(results_list, summary_data):

    out_dir = "/Users/salrei/Documents/Diss/Exp1/plots"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results_list)
    summary_df = pd.DataFrame(summary_data)

    QUALITY_METRICS = {
        "Proposed": "quality_nn",
        "SER-FIQ": "quality_serfiq",
        "Embedding Norm": "quality_norm",
        "Laplacian": "quality_lap"
    }

    DEG_TYPES = df["category"].unique()
    MARKERS = ['o', 's', '^', 'D']

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    fig.suptitle("Quality Score vs Face Similarity", fontweight="bold")

    for ax, (name, col) in zip(axes, QUALITY_METRICS.items()):

        all_q, all_s = [], []

        for i, deg in enumerate(DEG_TYPES):

            sub = df[df["category"] == deg]

            q = sub[col].values
            s = sub["actual_sim"].values

            ax.scatter(q, s,
                       label=deg,
                       marker=MARKERS[i % len(MARKERS)],
                       alpha=0.7)

            all_q.extend(q)
            all_s.extend(s)

        if len(all_q) > 3:

            rho, _ = spearmanr(all_q, all_s)

            z = np.polyfit(all_q, all_s, 1)
            xs = np.linspace(min(all_q), max(all_q), 100)

            ax.plot(xs, np.poly1d(z)(xs),
                    '--', color='black', alpha=0.5)

            ax.set_title(f"{name}\nρ={rho:.3f}")

        else:
            ax.set_title(name)

        ax.set_xlabel("Quality Score")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Similarity")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_quality_vs_similarity.png"), dpi=300)
    plt.close()


    fig, ax = plt.subplots(figsize=(8, 4))

    for deg in summary_df["Degradation"].unique():

        sub = summary_df[summary_df["Degradation"] == deg]

        ax.plot(range(len(sub["Level"])),
                sub["TAR@FAR=1e-3"],
                marker='o',
                label=deg)

    ax.set_title("TAR@FAR=1e-3 vs Degradation Level",
                 fontweight="bold")

    ax.set_xlabel("Level Index")
    ax.set_ylabel("TAR")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tar_vs_degradation.png"), dpi=300)
    plt.close()


    methods = {
        "Proposed": "quality_nn",
        "SER-FIQ": "quality_serfiq",
        "Embedding Norm": "quality_norm",
        "Laplacian": "quality_lap"
    }

    heat = []

    for deg in DEG_TYPES:

        sub = df[df["category"] == deg]

        row = []

        for m in methods.values():

            rho, _ = spearmanr(sub[m], sub["actual_sim"])
            row.append(rho)

        heat.append(row)

    heat = np.array(heat)

    fig, ax = plt.subplots(figsize=(7, 4))

    im = ax.imshow(heat, cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(list(methods.keys()), rotation=30)

    ax.set_yticks(range(len(DEG_TYPES)))
    ax.set_yticklabels(DEG_TYPES)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat[i, j]:.2f}",
                    ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax)

    ax.set_title("Spearman Correlation per Degradation",
                 fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spearman_heatmap.png"), dpi=300)
    plt.close()
    
    
def run_evaluation():
    device = torch.device("mps")

    clean_root = "/Users/salrei/Documents/Diss/Exp1/Dataset/lfw_orig"
    degraded_root = "/Users/salrei/Documents/Diss/Exp1/Dataset/lfw_new"

    model = get_model(os.path.join(model_path, "checkpoints/resnet50.pth"), device)

    categories = ['blur', 'noise', 'low_res', 'occlusion']

    all_clean_files = []
    for root, _, files in os.walk(clean_root):
        for f in files:
            if f.lower().endswith(('.jpg', '.png')) and not f.startswith('.'):
                all_clean_files.append(os.path.relpath(os.path.join(root, f), clean_root))

    test_files = all_clean_files[:19]

    print(f"Pre-processing {len(test_files)} clean images...")
    clean_data = {}

    for f in tqdm(test_files):
        img = cv2.imread(os.path.join(clean_root, f))
        if img is not None:
            clean_data[f] = compute_all_metrics(img, model, device)

    results_list = []
    summary_data = []

    print("\nStarting Degraded Evaluation...")

    for cat in categories:
        for lvl in range(1, 6):

            degraded_features = {}
            cat_metrics = []

            for f in test_files:
                path = os.path.join(degraded_root, cat, f"level_{lvl}", f)
                if not os.path.exists(path):
                    continue

                img = cv2.imread(path)
                if img is None:
                    continue

                m = compute_all_metrics(img, model, device)
                degraded_features[f] = m['feat']
                cat_metrics.append(m)

            if len(degraded_features) == 0:
                continue

            clean_feats = {k: v['feat'] for k, v in clean_data.items()}

            tar, auc_val = run_verification(clean_feats, degraded_features)
            avg_q = np.mean([x['q_nn'] for x in cat_metrics])

            print(f"{cat:10s} L{lvl}: TAR={tar:.4f} | Quality={avg_q:.4f}")

            for f, m in zip(degraded_features.keys(), cat_metrics):
                f1 = clean_data[f]['feat'].flatten()
                f2 = m['feat'].flatten()

                sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))

                results_list.append({
                    'category': cat,
                    'actual_sim': sim,
                    'quality_nn': m['q_nn'],
                    'quality_serfiq': m['serfiq'],
                    'quality_norm': m['norm'],
                    'quality_lap': m['lap']
                })

            summary_data.append({
                'Degradation': cat,
                'Level': f'L{lvl}',
                'TAR@FAR=1e-3': f"{tar:.4f}",
                'AUC': f"{auc_val:.4f}",
                'Quality(NN)': f"{avg_q:.4f}"
            })

    print("\n" + "="*70)
    print("FINAL TABLES")
    print("="*70)
    print(pd.DataFrame(summary_data).to_string(index=False))

    df = pd.DataFrame(results_list)

    df['quality_log'] = np.log1p(df['quality_nn'])
    df['serfiq_log'] = np.log1p(df['quality_serfiq'])
    df['sim_log'] = np.log1p(df['actual_sim'])

    METHODS = {
        'quality_log': 'Proposed (Improved)',
        'serfiq_log': 'SER-FIQ',
        'quality_norm': 'Embedding Norm',
        'quality_lap': 'Laplacian'
    }

    rows = []

    for cat in categories:
        sub = df[df['category'] == cat]
        row = {'Degradation': cat}

        for k, name in METHODS.items():
            rho, _ = spearmanr(sub[k], sub['sim_log'])
            row[name] = round(rho, 4)

        rows.append(row)

    print("\nTable — Spearman Correlation\n")
    print(pd.DataFrame(rows).set_index('Degradation'))
    
    print("\n" + "="*70)
    print("TABLE 2 — Overall Spearman ρ Comparison")
    print("="*70)

    overall_rows = []
    n_samples = len(df)

    for k, name in METHODS.items():
        rho, p_val = spearmanr(df[k], df['sim_log'])
        
        overall_rows.append({
            'Method': name,
            'Spearman ρ': round(rho, 4),
            'p-value': p_val,
            'n': n_samples
        })

    table2_df = pd.DataFrame(overall_rows).set_index('Method')
    print(table2_df.to_string())
    make_all_plots(results_list, summary_data)


if __name__ == "__main__":
    run_evaluation()
    