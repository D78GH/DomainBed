# diagnostics.py

import torch
import torch.nn.functional as F


class SemanticConsistencyTracker:

    def __init__(self, print_every=300):

        self.print_every = print_every

        self.step_count = 0

        self.history = []

    @torch.no_grad()
    def get_domain_class_prototypes(
        self,
        features,
        labels
    ):
        """
        Creates one class representation for each class
        present in a domain mini-batch.
        """

        features = F.normalize(
            features.detach(),
            dim=1
        )

        class_protos = {}

        for c in torch.unique(labels):

            mask = labels == c

            class_proto = features[mask].mean(0)

            class_proto = F.normalize(
                class_proto,
                dim=0
            )

            class_protos[int(c.item())] = (
                class_proto.detach()
            )

        return class_protos

    @torch.no_grad()
    def measure_batch_global_similarity(
        self,
        domain_class_protos,
        global_prototypes
    ):
        """
        Measures similarity between each domain's
        class representation and the global class prototype.
        """

        similarities = []

        for domain_protos in domain_class_protos:

            for c, batch_proto in domain_protos.items():

                if c >= global_prototypes.size(0):
                    continue

                global_proto = F.normalize(
                    global_prototypes[c].mean(0),
                    dim=0
                )

                similarity = torch.dot(
                    batch_proto,
                    global_proto
                )

                similarities.append(
                    similarity.item()
                )

        if len(similarities) == 0:
            return 0.0

        return sum(similarities) / len(similarities)

    @torch.no_grad()
    def measure_cross_domain_similarity(
        self,
        domain_class_protos
    ):
        """
        Measures similarity between representations of the
        SAME class across DIFFERENT source domains.
        """

        similarities = []

        num_domains = len(domain_class_protos)

        for i in range(num_domains):

            for j in range(i + 1, num_domains):

                domain_a = domain_class_protos[i]

                domain_b = domain_class_protos[j]

                shared_classes = (
                    set(domain_a.keys())
                    & set(domain_b.keys())
                )

                for c in shared_classes:

                    proto_a = domain_a[c]

                    proto_b = domain_b[c]

                    similarity = torch.dot(
                        proto_a,
                        proto_b
                    )

                    similarities.append(
                        similarity.item()
                    )

        if len(similarities) == 0:
            return 0.0

        return sum(similarities) / len(similarities)

    @torch.no_grad()
    def update(
        self,
        domain_features,
        domain_labels,
        global_prototypes,
        target_accuracy=None
    ):
        """
        domain_features:
            list of feature tensors, one per source domain

        domain_labels:
            list of label tensors, one per source domain

        global_prototypes:
            model's global class prototypes

        target_accuracy:
            Classification accuracy on the held-out
            target domain. This is diagnostic only and
            does not affect training.
        """

        self.step_count += 1

        # --------------------------------------------------
        # Create class representations for each domain
        # --------------------------------------------------

        domain_class_protos = []

        for features, labels in zip(
            domain_features,
            domain_labels
        ):

            class_protos = self.get_domain_class_prototypes(
                features,
                labels
            )

            domain_class_protos.append(
                class_protos
            )

        # --------------------------------------------------
        # Domain -> global prototype similarity
        # --------------------------------------------------

        batch_global_similarity = (
            self.measure_batch_global_similarity(
                domain_class_protos,
                global_prototypes
            )
        )

        # --------------------------------------------------
        # Same-class cross-domain similarity
        # --------------------------------------------------

        cross_domain_similarity = (
            self.measure_cross_domain_similarity(
                domain_class_protos
            )
        )

        # --------------------------------------------------
        # Store diagnostic results
        # --------------------------------------------------

        result = {
            "step":
                self.step_count,

            "batch_global_similarity":
                batch_global_similarity,

            "cross_domain_similarity":
                cross_domain_similarity,

            "target_accuracy":
                target_accuracy
        }

        self.history.append(result)

        # --------------------------------------------------
        # Print diagnostics
        # --------------------------------------------------

        if self.step_count % self.print_every == 0:

            print(
                "\n[SEMANTIC DIAGNOSTICS]"
            )

            print(
                f"Step: {self.step_count}"
            )

            print(
                "Domain -> Global Prototype Similarity: "
                f"{batch_global_similarity:.4f}"
            )

            print(
                "Cross-Domain Same-Class Similarity: "
                f"{cross_domain_similarity:.4f}"
            )

            if target_accuracy is not None:

                print(
                    "Held-Out Domain Accuracy: "
                    f"{target_accuracy:.4f}"
                )

        return result

    def get_history(self):

        return self.history