import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def _get_visualisation_path(output_dir, test_env, filename):
    visualisation_dir = os.path.join(output_dir, "visualisations", f"test_env_{test_env}")
    os.makedirs(visualisation_dir, exist_ok=True)
    return os.path.join(visualisation_dir, filename)

@torch.no_grad()
def _extract_features(model, x, batch_size=32):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    features = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size].to(device)
        batch_features = model.featurizer(batch)
        batch_features = F.normalize(batch_features, dim=1)
        features.append(batch_features.cpu())
        del batch
        del batch_features
    if not features:
        raise ValueError("No features could be extracted for visualisation.")
    features = torch.cat(features, dim=0)
    if was_training:
        model.train()
    return features

@torch.no_grad()
def prepare_prototype_pca(model, x, max_samples=500, batch_size=32):
    features = _extract_features(model, x, batch_size=batch_size)
    if len(features) > max_samples:
        idx = torch.randperm(len(features))[:max_samples]
        features = features[idx]
    prototypes = F.normalize(model.prototypes, dim=2).detach().cpu()
    prototype_flat = prototypes.reshape(-1, prototypes.shape[-1])
    combined = torch.cat([features, prototype_flat], dim=0).numpy()
    pca = PCA(n_components=2)
    pca.fit(combined)
    return pca

@torch.no_grad()
def plot_prototypes(model, x, y, pca, step=None, max_samples=500, batch_size=32, output_dir=".", test_env=0):
    was_training = model.training
    model.eval()
    if len(x) > max_samples:
        idx = torch.randperm(len(x))[:max_samples]
        x = x[idx]
        y = y[idx]
    features = _extract_features(model, x, batch_size=batch_size)
    prototypes = F.normalize(model.prototypes, dim=2).detach().cpu()
    prototype_flat = prototypes.reshape(-1, prototypes.shape[-1])
    sample_2d = pca.transform(features.numpy())
    prototype_2d = pca.transform(prototype_flat.numpy())
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(sample_2d[:, 0], sample_2d[:, 1], c=y.numpy(), cmap="tab10", alpha=0.35, s=25)
    for c in range(model.num_classes):
        for k in range(model.num_prototypes):
            idx = c * model.num_prototypes + k
            ax.scatter(prototype_2d[idx, 0], prototype_2d[idx, 1], marker="X", s=250, color=plt.cm.tab10(c), edgecolor="black", linewidth=1.5, zorder=10)
            ax.annotate(f"C{c}-P{k}", (prototype_2d[idx, 0], prototype_2d[idx, 1]), xytext=(6, 6), textcoords="offset points", fontsize=9, fontweight="bold")
    step_name = "final" if step is None else str(step)
    title = "MLPMCL Prototype Visualisation" if step is None else f"MLPMCL Prototype Visualisation - Step {step}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_path = _get_visualisation_path(output_dir, test_env, f"mlpmcl_prototypes_step_{step_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prototype visualisation to: {save_path}")
    if was_training:
        model.train()

@torch.no_grad()
def plot_domain_generalization(model, train_x, train_y, unseen_x, unseen_y, pca, step=None, max_samples=500, batch_size=32, output_dir=".", test_env=0):
    was_training = model.training
    model.eval()
    if len(train_x) > max_samples:
        idx = torch.randperm(len(train_x))[:max_samples]
        train_x = train_x[idx]
        train_y = train_y[idx]
    if len(unseen_x) > max_samples:
        idx = torch.randperm(len(unseen_x))[:max_samples]
        unseen_x = unseen_x[idx]
        unseen_y = unseen_y[idx]
    train_features = _extract_features(model, train_x, batch_size=batch_size)
    unseen_features = _extract_features(model, unseen_x, batch_size=batch_size)
    prototypes = F.normalize(model.prototypes, dim=2).detach().cpu()
    prototype_flat = prototypes.reshape(-1, prototypes.shape[-1])
    train_2d = pca.transform(train_features.numpy())
    unseen_2d = pca.transform(unseen_features.numpy())
    prototype_2d = pca.transform(prototype_flat.numpy())
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(train_2d[:, 0], train_2d[:, 1], c=train_y.numpy(), cmap="tab10", alpha=0.20, s=25, marker="o", label="Training domain")
    ax.scatter(unseen_2d[:, 0], unseen_2d[:, 1], c=unseen_y.numpy(), cmap="tab10", alpha=0.65, s=35, marker="^", edgecolor="black", linewidth=0.3, label="Unseen domain")
    for c in range(model.num_classes):
        for k in range(model.num_prototypes):
            idx = c * model.num_prototypes + k
            ax.scatter(prototype_2d[idx, 0], prototype_2d[idx, 1], marker="X", s=280, color=plt.cm.tab10(c), edgecolor="black", linewidth=1.5, zorder=10)
            ax.annotate(f"C{c}-P{k}", (prototype_2d[idx, 0], prototype_2d[idx, 1]), xytext=(6, 6), textcoords="offset points", fontsize=8, fontweight="bold")
    step_name = "final" if step is None else str(step)
    title = "MLPMCL: Training vs Unseen-Domain Representations" if step is None else f"MLPMCL: Training vs Unseen-Domain Representations - Step {step}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_path = _get_visualisation_path(output_dir, test_env, f"mlpmcl_domain_generalization_step_{step_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved domain generalization visualisation to: {save_path}")
    if was_training:
        model.train()

def plot_learning_dynamics(history, output_dir=".", test_env=0):
    if not history:
        return
    steps = [x["step"] for x in history]
    fig, ax = plt.subplots(figsize=(10, 6))
    if any(x.get("loss") is not None for x in history):
        ax.plot(steps, [x["loss"] if x.get("loss") is not None else float("nan") for x in history], label="Total loss", linewidth=2)
    if any(x.get("ce_loss") is not None for x in history):
        ax.plot(steps, [x["ce_loss"] if x.get("ce_loss") is not None else float("nan") for x in history], label="Classification loss")
    if any(x.get("proto_loss") is not None for x in history):
        ax.plot(steps, [x["proto_loss"] if x.get("proto_loss") is not None else float("nan") for x in history], label="Prototype contrastive loss")
    if any(x.get("mem_loss") is not None for x in history):
        ax.plot(steps, [x["mem_loss"] if x.get("mem_loss") is not None else float("nan") for x in history], label="Memory alignment loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("MLPMCL Learning Dynamics")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save_path = _get_visualisation_path(output_dir, test_env, "mlpmcl_learning_dynamics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning dynamics visualisation to: {save_path}")