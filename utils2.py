import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score


# =========================================================
# GENERAL UTILITIES
# =========================================================

def unfold_label(labels, classes):
    """
    Convert integer labels into one-hot labels.
    """
    labels = np.asarray(labels)

    assert len(np.unique(labels)) == classes

    min_label = np.min(labels)
    one_hot = np.zeros((len(labels), classes), dtype=np.int8)

    for i, label in enumerate(labels):
        one_hot[i, int(label) - min_label] = 1

    return one_hot


def shuffle_data(samples, labels):
    idx = np.random.permutation(len(labels))
    return samples[idx], labels[idx]


def shuffle_list(items):
    np.random.shuffle(items)
    return items


def shuffle_list_with_ind(items):
    idx = np.random.permutation(len(items))
    return items[idx], idx


def sgd(parameters,
        lr,
        weight_decay=5e-4,
        momentum=0.9):
    return optim.SGD(
        parameters,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def fix_seed(seed=42):
    """
    Fully deterministic training.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_log(log, log_path):
    with open(log_path, "a") as f:
        f.write(str(log))
        f.write("\n")


def compute_accuracy(predictions, labels):
    """
    Expects one-hot labels and class probabilities/logits.
    """
    return accuracy_score(
        np.argmax(labels, axis=-1),
        np.argmax(predictions, axis=-1),
    )


# =========================================================
# DOMAIN-GENERALISED PROTOTYPE MEMORY
# =========================================================

class DGPrototypeMemory:
    """
    Stores class prototypes using EMA updates and
    maintains a short history for invariance regularization.
    """

    def __init__(self,
                 momentum=0.9,
                 max_history=10):
        self.prototypes = {}     # cls -> EMA prototype
        self.history = {}        # cls -> [historical prototypes]

        self.momentum = momentum
        self.max_history = max_history

    @torch.no_grad()
    def update(self, batch_protos):
        """
        Args:
            batch_protos:
                dict[class_id] -> prototype tensor [D]
        """
        for cls, proto in batch_protos.items():

            proto = proto.detach().cpu()

            # EMA prototype update
            if cls not in self.prototypes:
                self.prototypes[cls] = proto
            else:
                self.prototypes[cls] = (
                    self.momentum * self.prototypes[cls]
                    + (1.0 - self.momentum) * proto
                )

            # Prototype history
            self.history.setdefault(cls, []).append(proto)

            if len(self.history[cls]) > self.max_history:
                self.history[cls].pop(0)

    def get_prototypes_tensor(self, device):
        """
        Returns:
            Tensor [num_classes, D]
            or None if memory is empty.
        """
        if len(self.prototypes) == 0:
            return None

        return torch.stack(
            list(self.prototypes.values())
        ).to(device)

    def get_class_prototype(self, cls, device):
        if cls not in self.prototypes:
            return None

        return self.prototypes[cls].to(device)

    def __len__(self):
        return len(self.prototypes)


# =========================================================
# DG LOSSES
# =========================================================

def class_prototype_contrastive_loss(
    features,
    labels,
    prototype_memory,
    temperature=0.07,
):
    """
    Contrastive alignment between sample features and
    class prototypes stored in memory.

    features: [B, D]
    labels:   [B]
    """

    if len(prototype_memory.prototypes) == 0:
        return torch.tensor(
            0.0,
            device=features.device,
            dtype=features.dtype,
        )

    # Normalize features
    features = F.normalize(features, dim=1)

    # Build prototype matrix and mapping
    prototype_classes = sorted(
        prototype_memory.prototypes.keys()
    )

    prototypes = torch.stack(
        [
            prototype_memory.prototypes[c].to(features.device)
            for c in prototype_classes
        ],
        dim=0,
    )

    prototypes = F.normalize(prototypes, dim=1)

    # [B, num_prototypes]
    logits = (
        features @ prototypes.T
    ) / temperature

    # Map actual class labels -> prototype column indices
    class_to_index = {
        cls: idx
        for idx, cls in enumerate(prototype_classes)
    }

    valid_mask = torch.tensor(
        [int(lbl.item()) in class_to_index for lbl in labels],
        device=features.device,
        dtype=torch.bool,
    )

    # No classes in memory match this batch
    if valid_mask.sum() == 0:
        return torch.tensor(
            0.0,
            device=features.device,
            dtype=features.dtype,
        )

    logits = logits[valid_mask]

    target = torch.tensor(
        [
            class_to_index[int(lbl.item())]
            for lbl in labels[valid_mask]
        ],
        device=features.device,
        dtype=torch.long,
    )

    return F.cross_entropy(logits, target)

def prototype_invariance_loss(memory: DGPrototypeMemory):
    """
    Encourages historical prototypes of the same class
    to remain consistent across domains/iterations.
    """
    losses = []

    for _, history in memory.history.items():

        if len(history) < 2:
            continue

        stack = torch.stack(history)

        for i in range(len(stack) - 1):
            losses.append(
                1.0 - F.cosine_similarity(
                    stack[i],
                    stack[i + 1],
                    dim=0,
                )
            )

    if len(losses) == 0:
        return torch.tensor(0.0)

    return torch.stack(losses).mean()


def prototype_diversity_loss(prototypes):
    """
    Encourages prototypes from different classes
    to be dissimilar.
    """
    if prototypes.shape[0] < 2:
        return torch.tensor(
            0.0,
            device=prototypes.device,
        )

    prototypes = F.normalize(prototypes, dim=1)

    similarity = prototypes @ prototypes.T

    off_diag = (
        similarity.sum()
        - similarity.diag().sum()
    )

    return off_diag / (prototypes.shape[0] ** 2)


def memory_alignment_loss(batch_protos,
                          memory: DGPrototypeMemory,
                          device):
    """
    Align current batch prototypes with memory prototypes.
    """
    if len(memory.prototypes) == 0:
        return torch.tensor(0.0, device=device)

    losses = []

    for cls, proto in batch_protos.items():

        if cls not in memory.prototypes:
            continue

        memory_proto = memory.prototypes[cls].to(device)

        losses.append(
            1.0 - F.cosine_similarity(
                proto,
                memory_proto,
                dim=0,
            )
        )

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def total_dg_loss(
    class_prototype_contrastive_loss,
    memory_align,
    invariance_loss,
    diversity_loss,
    w1=1.0,
    w2=0.1,
    w3=0.1,
    w4=0.05,
):
    """
    Final DG objective.
    """
    return (
        w1 * class_prototype_contrastive_loss
        + w2 * memory_align
        + w3 * invariance_loss
        + w4 * diversity_loss
    )