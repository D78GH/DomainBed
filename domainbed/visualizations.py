import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score

def get_visualisation_path(output_dir,test_env,filename):
    visualisation_dir=os.path.join(output_dir,"visualisations",f"test_env_{test_env}")
    os.makedirs(visualisation_dir,exist_ok=True)
    return os.path.join(visualisation_dir,filename)

def get_class_colors(num_classes):
    return [plt.cm.tab10(c%10) for c in range(num_classes)]

def get_model_network(model):
    # MLPMCL: featurizer/prototypes are directly on the algorithm.
    if hasattr(model,"featurizer") and hasattr(model,"prototypes") and hasattr(model,"num_prototypes"):
        return model
    # FishMLPMCL: featurizer/prototypes are inside model.network.
    if hasattr(model,"network"):
        network=model.network
        if hasattr(network,"featurizer") and hasattr(network,"prototypes") and hasattr(network,"num_prototypes"):
            return network
    raise AttributeError(
        f"{type(model).__name__} does not expose the required "
        "featurizer/prototypes/num_prototypes interface."
    )

def get_featurizer(model):
    return get_model_network(model).featurizer

def get_prototypes(model):
    return get_model_network(model).prototypes

def get_num_classes(model):
    if hasattr(model,"num_classes"):
        return model.num_classes
    network=get_model_network(model)
    if hasattr(network,"num_classes"):
        return network.num_classes
    raise AttributeError(f"Could not determine num_classes for {type(model).__name__}")

def get_num_prototypes(model):
    network=get_model_network(model)
    if hasattr(network,"num_prototypes"):
        return network.num_prototypes
    if hasattr(model,"num_prototypes"):
        return model.num_prototypes
    raise AttributeError(f"Could not determine num_prototypes for {type(model).__name__}")

@torch.no_grad()
def extract_features(model,x,batch_size=32):
    was_training=model.training
    model.eval()
    device=next(model.parameters()).device
    featurizer=get_featurizer(model)
    features=[]
    for i in range(0,len(x),batch_size):
        batch=x[i:i+batch_size].to(device)
        batch_features=F.normalize(featurizer(batch),dim=1)
        features.append(batch_features.cpu())
        del batch,batch_features
    if not features:
        raise ValueError("No features could be extracted for visualisation.")
    features=torch.cat(features,dim=0)
    if was_training:
        model.train()
    return features

@torch.no_grad()
def get_prototype_assignments(model,x,batch_size=32):
    features=extract_features(model,x,batch_size=batch_size)
    prototypes=F.normalize(get_prototypes(model),dim=2).detach().cpu()
    prototype_flat=prototypes.reshape(-1,prototypes.shape[-1])
    similarity=features@prototype_flat.T
    assignments=similarity.argmax(dim=1)
    return features,assignments,prototype_flat

@torch.no_grad()
def compute_prototype_mi(model,x,y,batch_size=32):
    _,assignments,_=get_prototype_assignments(model,x,batch_size=batch_size)
    return mutual_info_score(y.cpu().numpy(),assignments.numpy())

@torch.no_grad()
def prepare_prototype_pca(model,x,max_samples=500,batch_size=32):
    features=extract_features(model,x,batch_size=batch_size)
    if len(features)>max_samples:
        idx=torch.randperm(len(features))[:max_samples]
        features=features[idx]
    prototypes=F.normalize(get_prototypes(model),dim=2).detach().cpu()
    prototype_flat=prototypes.reshape(-1,prototypes.shape[-1])
    combined=torch.cat([features,prototype_flat],dim=0).numpy()
    pca=PCA(n_components=2)
    pca.fit(combined)
    return pca

@torch.no_grad()
def plot_prototypes(model,x,y,pca,step=None,max_samples=500,batch_size=32,output_dir=".",test_env=0):
    was_training=model.training
    model.eval()
    if len(x)>max_samples:
        idx=torch.randperm(len(x))[:max_samples]
        x=x[idx]
        y=y[idx]
    features=extract_features(model,x,batch_size=batch_size)
    prototypes=F.normalize(get_prototypes(model),dim=2).detach().cpu()
    prototype_flat=prototypes.reshape(-1,prototypes.shape[-1])
    sample_2d=pca.transform(features.numpy())
    prototype_2d=pca.transform(prototype_flat.numpy())
    num_classes=get_num_classes(model)
    num_prototypes=get_num_prototypes(model)
    prototype_markers=["X","P","D","*","^","s","v","<",">","p"]
    class_colors=get_class_colors(num_classes)
    fig,ax=plt.subplots(figsize=(16,8))
    sample_colors=[class_colors[int(label)%len(class_colors)] for label in y.numpy()]
    ax.scatter(sample_2d[:,0],sample_2d[:,1],c=sample_colors,alpha=1.0,s=32,marker="o",edgecolor="none",linewidths=0,zorder=1)
    for c in range(num_classes):
        for k in range(num_prototypes):
            idx=c*num_prototypes+k
            marker=prototype_markers[k%len(prototype_markers)]
            ax.scatter(prototype_2d[idx,0],prototype_2d[idx,1],marker=marker,s=420,color=class_colors[c],alpha=1.0,edgecolor="black",linewidth=2,zorder=10)
    class_handles=[ax.scatter([],[],marker="o",s=100,color=class_colors[c],alpha=1.0,label=f"Class {c}") for c in range(num_classes)]
    prototype_handles=[ax.scatter([],[],marker=prototype_markers[k%len(prototype_markers)],s=180,color="white",alpha=1.0,edgecolor="black",linewidth=2,label=f"P{k+1}") for k in range(num_prototypes)]
    legend1=ax.legend(handles=class_handles,title="Classes",loc="upper right",frameon=True)
    ax.add_artist(legend1)
    ax.legend(handles=prototype_handles,title="Prototype index",loc="lower right",frameon=True)
    step_name="final" if step is None else str(step)
    ax.set_title("MLPMCL Prototype Visualisation" if step is None else f"MLPMCL Prototype Visualisation - Step {step}",fontsize=16,fontweight="bold")
    ax.set_xlabel("PCA component 1",fontsize=12)
    ax.set_ylabel("PCA component 2",fontsize=12)
    ax.grid(True,color="gray",alpha=1.0,linewidth=0.5)
    ax.set_axisbelow(True)
    x_min,x_max=ax.get_xlim()
    x_center=(x_min+x_max)/2
    x_half=(x_max-x_min)/2
    ax.set_xlim(x_center-1.5*x_half,x_center+1.5*x_half)
    plt.tight_layout()
    save_path=get_visualisation_path(output_dir,test_env,f"mlpmcl_prototypes_step_{step_name}.png")
    plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prototype visualisation to: {save_path}")
    if was_training:
        model.train()

@torch.no_grad()
def plot_domain_generalization(model,train_x,train_y,unseen_x,unseen_y,pca,step=None,max_samples=500,batch_size=32,output_dir=".",test_env=0):
    was_training=model.training
    model.eval()
    if len(train_x)>max_samples:
        idx=torch.randperm(len(train_x))[:max_samples]
        train_x=train_x[idx]
        train_y=train_y[idx]
    if len(unseen_x)>max_samples:
        idx=torch.randperm(len(unseen_x))[:max_samples]
        unseen_x=unseen_x[idx]
        unseen_y=unseen_y[idx]
    train_features=extract_features(model,train_x,batch_size=batch_size)
    unseen_features=extract_features(model,unseen_x,batch_size=batch_size)
    prototypes=F.normalize(get_prototypes(model),dim=2).detach().cpu()
    prototype_flat=prototypes.reshape(-1,prototypes.shape[-1])
    train_2d=pca.transform(train_features.numpy())
    unseen_2d=pca.transform(unseen_features.numpy())
    prototype_2d=pca.transform(prototype_flat.numpy())
    num_classes=get_num_classes(model)
    num_prototypes=get_num_prototypes(model)
    prototype_markers=["X","P","D","*","s","v","<",">","p","h"]
    class_colors=get_class_colors(num_classes)
    fig,ax=plt.subplots(figsize=(16,8))
    train_colors=[class_colors[int(label)%len(class_colors)] for label in train_y.cpu().numpy()]
    unseen_colors=[class_colors[int(label)%len(class_colors)] for label in unseen_y.cpu().numpy()]
    ax.scatter(train_2d[:,0],train_2d[:,1],c=train_colors,alpha=1.0,s=30,marker="o",edgecolor="none",linewidths=0,zorder=1)
    ax.scatter(unseen_2d[:,0],unseen_2d[:,1],c=unseen_colors,alpha=1.0,s=42,marker="^",edgecolor="black",linewidth=0.5,zorder=2)
    for c in range(num_classes):
        for k in range(num_prototypes):
            idx=c*num_prototypes+k
            marker=prototype_markers[k%len(prototype_markers)]
            ax.scatter(prototype_2d[idx,0],prototype_2d[idx,1],marker=marker,s=420,color=class_colors[c],alpha=1.0,edgecolor="black",linewidth=2,zorder=10)
    class_handles=[ax.scatter([],[],marker="o",s=100,color=class_colors[c],alpha=1.0,label=f"Class {c}") for c in range(num_classes)]
    domain_handles=[ax.scatter([],[],marker="o",s=100,color="gray",alpha=1.0,label="Training domain"),ax.scatter([],[],marker="^",s=100,color="gray",alpha=1.0,edgecolor="black",label="Unseen domain")]
    prototype_handles=[ax.scatter([],[],marker=prototype_markers[k%len(prototype_markers)],s=180,color="white",alpha=1.0,edgecolor="black",linewidth=2,label=f"P{k+1}") for k in range(num_prototypes)]
    legend1=ax.legend(handles=class_handles,title="Classes",loc="upper right",frameon=True)
    ax.add_artist(legend1)
    legend2=ax.legend(handles=domain_handles,title="Domain",loc="lower left",frameon=True)
    ax.add_artist(legend2)
    ax.legend(handles=prototype_handles,title="Prototype index",loc="lower right",frameon=True)
    step_name="final" if step is None else str(step)
    ax.set_title("MLPMCL: Training vs Unseen-Domain Representations" if step is None else f"MLPMCL: Training vs Unseen-Domain Representations - Step {step}",fontsize=16,fontweight="bold")
    ax.set_xlabel("PCA component 1",fontsize=12)
    ax.set_ylabel("PCA component 2",fontsize=12)
    ax.grid(True,color="gray",alpha=1.0,linewidth=0.5)
    ax.set_axisbelow(True)
    x_min,x_max=ax.get_xlim()
    x_center=(x_min+x_max)/2
    x_half=(x_max-x_min)/2
    ax.set_xlim(x_center-1.5*x_half,x_center+1.5*x_half)
    plt.tight_layout()
    save_path=get_visualisation_path(output_dir,test_env,f"mlpmcl_domain_generalization_step_{step_name}.png")
    plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved domain generalization visualisation to: {save_path}")
    if was_training:
        model.train()

@torch.no_grad()
def plot_prototype_utilisation(model,x,y,step=None,max_samples=2000,batch_size=32,output_dir=".",test_env=0):
    if len(x)>max_samples:
        idx=torch.randperm(len(x))[:max_samples]
        x=x[idx]
        y=y[idx]
    _,assignments,_=get_prototype_assignments(model,x,batch_size=batch_size)
    num_classes=get_num_classes(model)
    num_prototypes=get_num_prototypes(model)
    n_prototypes=num_classes*num_prototypes
    counts=torch.bincount(assignments,minlength=n_prototypes).numpy()
    labels=[f"C{c}-P{k+1}" for c in range(num_classes) for k in range(num_prototypes)]
    class_colors=get_class_colors(num_classes)
    colors=[class_colors[c] for c in range(num_classes) for _ in range(num_prototypes)]
    fig,ax=plt.subplots(figsize=(12,6))
    ax.bar(np.arange(n_prototypes),counts,color=colors,edgecolor="black",linewidth=0.7)
    ax.set_xticks(np.arange(n_prototypes))
    ax.set_xticklabels(labels,rotation=45,ha="right")
    ax.set_xlabel("Prototype")
    ax.set_ylabel("Number of assigned samples")
    ax.set_title("MLPMCL Prototype Utilisation")
    ax.grid(axis="y",color="gray",alpha=1.0,linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    step_name="final" if step is None else str(step)
    save_path=get_visualisation_path(output_dir,test_env,f"mlpmcl_prototype_utilisation_step_{step_name}.png")
    plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prototype utilisation visualisation to: {save_path}")

@torch.no_grad()
def plot_prototype_class_heatmap(model,x,y,step=None,max_samples=2000,batch_size=32,output_dir=".",test_env=0):
    if len(x)>max_samples:
        idx=torch.randperm(len(x))[:max_samples]
        x=x[idx]
        y=y[idx]
    _,assignments,_=get_prototype_assignments(model,x,batch_size=batch_size)
    assignments=assignments.numpy()
    labels=y.cpu().numpy()
    num_classes=get_num_classes(model)
    num_prototypes=get_num_prototypes(model)
    n_prototypes=num_classes*num_prototypes
    matrix=np.zeros((n_prototypes,num_classes),dtype=np.float32)
    for p in range(n_prototypes):
        mask=assignments==p
        if mask.sum()>0:
            for c in range(num_classes):
                matrix[p,c]=np.mean(labels[mask]==c)
    prototype_labels=[f"C{c}-P{k+1}" for c in range(num_classes) for k in range(num_prototypes)]
    fig,ax=plt.subplots(figsize=(10,max(6,n_prototypes*0.35)))
    im=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels([f"Class {c}" for c in range(num_classes)])
    ax.set_yticks(np.arange(n_prototypes))
    ax.set_yticklabels(prototype_labels)
    for i in range(n_prototypes):
        for j in range(num_classes):
            value=matrix[i,j]
            ax.text(j,i,f"{value:.2f}",ha="center",va="center",color="white" if value>0.5 else "black",fontsize=9)
    ax.set_xlabel("Ground-truth class")
    ax.set_ylabel("Prototype")
    ax.set_title("MLPMCL Prototype-Class Semantic Alignment")
    cbar=fig.colorbar(im,ax=ax)
    cbar.set_label("Class proportion")
    plt.tight_layout()
    step_name="final" if step is None else str(step)
    save_path=get_visualisation_path(output_dir,test_env,f"mlpmcl_prototype_class_heatmap_step_{step_name}.png")
    plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prototype-class heatmap to: {save_path}")

def plot_prototype_mutual_information(history,output_dir=".",test_env=0):
    valid=[h for h in history if h.get("prototype_mi") is not None]
    if not valid:
        return
    steps=[h["step"] for h in valid]
    mi=[h["prototype_mi"] for h in valid]
    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(steps,mi,marker="o",linewidth=2,color="#E76F51",alpha=1.0,label="MLPMCL prototypes")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mutual information with class labels")
    ax.set_title("MLPMCL Prototype-Class Mutual Information")
    ax.grid(True,color="gray",alpha=1.0,linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    save_path=get_visualisation_path(output_dir,test_env,"mlpmcl_prototype_mutual_information.png")
    plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prototype mutual information visualisation to: {save_path}")

def plot_learning_dynamics(history,output_dir=".",test_env=0):
    if not history:
        return
    steps=[x["step"] for x in history]
    fig,ax=plt.subplots(figsize=(10,6))
    if any(x.get("loss") is not None for x in history):
        ax.plot(steps,[x["loss"] if x.get("loss") is not None else float("nan") for x in history],label="Total loss",linewidth=2,alpha=1.0)
    if any(x.get("ce_loss") is not None for x in history):
        ax.plot(steps,[x["ce_loss"] if x.get("ce_loss") is not None else float("nan") for x in history],label="Classification loss",alpha=1.0)
    if any(x.get("proto_loss") is not None for x in history):
        ax.plot(steps,[x["proto_loss"] if x.get("proto_loss") is not None else float("nan") for x in history],label="Prototype contrastive loss",alpha=1.0)
    if any(x.get("mem_loss") is not None for x in history):
        ax.plot(steps,[x["mem_loss"] if x.get("mem_loss") is not None else float("nan") for x in history],label="Memory alignment loss",alpha=1.0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("MLPMCL Learning Dynamics")
    ax.legend()
    ax.grid(True,color="gray",alpha=1.0,linewidth=0.5)
    plt.tight_layout()
    save_path=get_visualisation_path(output_dir,test_env,"mlpmcl_learning_dynamics.png")
    plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning dynamics visualisation to: {save_path}")

def plot_all_prototype_visualisations(model,x,y,history=None,pca=None,step=None,max_samples=500,batch_size=32,output_dir=".",test_env=0):
    if pca is None:
        pca=prepare_prototype_pca(model,x,max_samples=max_samples,batch_size=batch_size)
    plot_prototypes(model,x,y,pca,step=step,max_samples=max_samples,batch_size=batch_size,output_dir=output_dir,test_env=test_env)
    plot_prototype_utilisation(model,x,y,step=step,max_samples=max_samples,batch_size=batch_size,output_dir=output_dir,test_env=test_env)
    plot_prototype_class_heatmap(model,x,y,step=step,max_samples=max_samples,batch_size=batch_size,output_dir=output_dir,test_env=test_env)
    if history is not None:
        plot_prototype_mutual_information(history,output_dir=output_dir,test_env=test_env)
        plot_learning_dynamics(history,output_dir=output_dir,test_env=test_env)