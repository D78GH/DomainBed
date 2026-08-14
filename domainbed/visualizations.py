import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def _get_visualisation_path(filename):
    visualisation_dir = os.path.join(os.getcwd(), "visualisations")
    os.makedirs(visualisation_dir, exist_ok=True)
    return os.path.join(visualisation_dir, filename)

@torch.no_grad()
def prepare_prototype_pca(model, x, max_samples=1000):
    was_training = model.training
    model.eval()
    features = F.normalize(model.featurizer(x), dim=1)
    if len(features) > max_samples:
        idx = torch.randperm(len(features), device=features.device)[:max_samples]
        features = features[idx]
    prototypes = F.normalize(model.prototypes, dim=2)
    prototype_flat = prototypes.reshape(-1, prototypes.shape[-1])
    combined = torch.cat([features, prototype_flat], dim=0).cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(combined)
    if was_training:
        model.train()
    return pca

@torch.no_grad()
def plot_prototypes(model, x, y, pca, step=None, max_samples=1000):
    was_training = model.training
    model.eval()
    features = F.normalize(model.featurizer(x), dim=1)
    if len(features) > max_samples:
        idx = torch.randperm(len(features), device=features.device)[:max_samples]
        features = features[idx]
        y = y[idx]
    prototypes = F.normalize(model.prototypes, dim=2)
    prototype_flat = prototypes.reshape(-1, prototypes.shape[-1])
    sample_2d = pca.transform(features.cpu().numpy())
    prototype_2d = pca.transform(prototype_flat.cpu().numpy())
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(sample_2d[:, 0], sample_2d[:, 1], c=y.cpu().numpy(), cmap="tab10", alpha=0.35, s=25)
    for c in range(model.num_classes):
        for k in range(model.num_prototypes):
            idx = c * model.num_prototypes + k
            ax.scatter(prototype_2d[idx, 0], prototype_2d[idx, 1], marker="X", s=250, color=plt.cm.tab10(c), edgecolor="black", linewidth=1.5, zorder=10)
            ax.annotate(f"C{c}-P{k}", (prototype_2d[idx, 0], prototype_2d[idx, 1]), xytext=(6, 6), textcoords="offset points", fontsize=9, fontweight="bold")
    step_name = "final" if step is None else str(step)
    ax.set_title("MLPMCL Prototype Visualisation" if step is None else f"MLPMCL Prototype Visualisation - Step {step}", fontsize=14)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_path = _get_visualisation_path(f"mlpmcl_prototypes_step_{step_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prototype visualisation to: {save_path}")
    if was_training:
        model.train()

@torch.no_grad()
def plot_domain_generalization(model, train_x, train_y, unseen_x, unseen_y, pca, step=None, max_samples=500):
    was_training = model.training
    model.eval()
    train_features = F.normalize(model.featurizer(train_x), dim=1)
    unseen_features = F.normalize(model.featurizer(unseen_x), dim=1)
    if len(train_features) > max_samples:
        idx = torch.randperm(len(train_features), device=train_features.device)[:max_samples]
        train_features = train_features[idx]
        train_y = train_y[idx]
    if len(unseen_features) > max_samples:
        idx = torch.randperm(len(unseen_features), device=unseen_features.device)[:max_samples]
        unseen_features = unseen_features[idx]
        unseen_y = unseen_y[idx]
    prototypes = F.normalize(model.prototypes, dim=2)
    prototype_flat = prototypes.reshape(-1, prototypes.shape[-1])
    train_2d = pca.transform(train_features.cpu().numpy())
    unseen_2d = pca.transform(unseen_features.cpu().numpy())
    prototype_2d = pca.transform(prototype_flat.cpu().numpy())
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(train_2d[:, 0], train_2d[:, 1], c=train_y.cpu().numpy(), cmap="tab10", alpha=0.20, s=25, marker="o", label="Training domain")
    ax.scatter(unseen_2d[:, 0], unseen_2d[:, 1], c=unseen_y.cpu().numpy(), cmap="tab10", alpha=0.65, s=35, marker="^", edgecolor="black", linewidth=0.3, label="Unseen domain")
    for c in range(model.num_classes):
        for k in range(model.num_prototypes):
            idx = c * model.num_prototypes + k
            ax.scatter(prototype_2d[idx, 0], prototype_2d[idx, 1], marker="X", s=280, color=plt.cm.tab10(c), edgecolor="black", linewidth=1.5, zorder=10)
            ax.annotate(f"C{c}-P{k}", (prototype_2d[idx, 0], prototype_2d[idx, 1]), xytext=(6, 6), textcoords="offset points", fontsize=8, fontweight="bold")
    step_name = "final" if step is None else str(step)
    ax.set_title("MLPMCL: Training vs Unseen-Domain Representations" if step is None else f"MLPMCL: Training vs Unseen-Domain Representations - Step {step}", fontsize=14)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_path = _get_visualisation_path(f"mlpmcl_domain_generalization_step_{step_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved domain generalization visualisation to: {save_path}")
    if was_training:
        model.train()

def plot_learning_dynamics(history):
    if not history:
        return
    steps = [x["step"] for x in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    if "loss" in history[0]:
        ax.plot(steps, [x["loss"] for x in history], label="Total loss", linewidth=2)
    if "ce_loss" in history[0]:
        ax.plot(steps, [x["ce_loss"] for x in history], label="Classification loss")
    if "proto_loss" in history[0]:
        ax.plot(steps, [x["proto_loss"] for x in history], label="Prototype contrastive loss")
    if "mem_loss" in history[0]:
        ax.plot(steps, [x["mem_loss"] for x in history], label="Memory alignment loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("MLPMCL Learning Dynamics")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_path = _get_visualisation_path("mlpmcl_learning_dynamics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning dynamics visualisation to: {save_path}")