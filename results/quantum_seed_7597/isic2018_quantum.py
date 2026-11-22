"""ISIC 2018 Task 3 lifelong learning + hybrid quantum-classical classification.

Standalone Python script converted from the notebook workflow.
Before running: set SEED to the last four digits of your applicant reference.
Uses a local ISIC Task 3 dataset. Internet is only needed to fetch pretrained ResNet-18 weights
(unless the weights are already cached). See DATA_DIR below.
"""
import os, glob, random, time, json, copy, zipfile, urllib.request, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import torchvision
from torchvision import transforms, models
import pennylane as qml

# Prefer CUDA when the installed PyTorch build can access the NVIDIA GPU.
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
print("Device:", DEVICE)
if CUDA_AVAILABLE:
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA is not available in this PyTorch environment.")
    print("If your PC has an NVIDIA GPU, install CUDA-enabled PyTorch in this env:")
    print("python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
# COURSEWORK REQUIREMENT 2: Applicant-reference last four digits are the seed.
# The seed deterministically shuffles the seven classes and controls the stratified split.
SEED = 7597  # Applicant-reference seed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
# COURSEWORK REQUIREMENT 1: Create a temporal/lifelong dataset from the local ISIC 2018 Task 3
# classification dataset. Set DATA_DIR to the folder containing extracted images/labels,
# or to the folder containing the two downloaded ZIP files.
# Dataset discovery: allow the dataset beside the script, in the current working
# folder, or in a nearby parent folder. Set ISIC_DATA_DIR to override this.
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_DATA_DIR = os.environ.get("ISIC_DATA_DIR")
candidate_roots = []
if ENV_DATA_DIR:
    candidate_roots.append(Path(ENV_DATA_DIR).expanduser().resolve())
candidate_roots.extend([
    SCRIPT_DIR / "isic2018_task3",
    Path.cwd() / "isic2018_task3",
    SCRIPT_DIR,
    Path.cwd(),
    SCRIPT_DIR.parent / "isic2018_task3",
    Path.cwd().parent / "isic2018_task3",
])
# Remove duplicates while preserving candidate order.
unique_roots = []
seen_roots = set()
for root in candidate_roots:
    root = root.resolve()
    if root not in seen_roots:
        unique_roots.append(root)
        seen_roots.add(root)

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

def find_images(root):
    found = []
    if root.exists():
        for pattern in IMAGE_EXTENSIONS:
            found.extend(root.rglob(pattern))
    return sorted({str(p.resolve()) for p in found})

def find_label_csvs(root):
    if not root.exists():
        return []
    return sorted({str(p.resolve()) for p in root.rglob("*.csv")})

# Choose the candidate root that contains the greatest number of image files
# and at least one label CSV. This avoids relying on the terminal's cwd alone.
root_scores = []
image_paths = []
csv_paths = []
for root in unique_roots:
    imgs = find_images(root)
    csvs = find_label_csvs(root)
    if imgs and csvs:
        root_scores.append((len(imgs), root, imgs, csvs))
if not root_scores:
    DATA_DIR = (Path(ENV_DATA_DIR).expanduser().resolve() if ENV_DATA_DIR
                else SCRIPT_DIR / "isic2018_task3")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
else:
    root_scores.sort(key=lambda item: item[0], reverse=True)
    _, DATA_DIR, image_paths, csv_paths = root_scores[0]

# Standard locations for optional ZIP extraction.
img_extract = DATA_DIR / "images"
label_extract = DATA_DIR / "labels"
img_extract.mkdir(parents=True, exist_ok=True)
label_extract.mkdir(parents=True, exist_ok=True)

def extract_if_needed(zip_path, destination):
    zip_path = Path(zip_path)
    if zip_path.exists() and not find_images(destination) and not find_label_csvs(destination):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(destination)

# If only ZIPs are present, extract them locally.
if not image_paths:
    local_img_zip = next(iter(DATA_DIR.glob("*Task3*Training*Input*.zip")), None)
    if local_img_zip:
        extract_if_needed(local_img_zip, img_extract)
if not csv_paths:
    local_label_zip = next(iter(DATA_DIR.glob("*Task3*GroundTruth*.zip")), None)
    if local_label_zip:
        extract_if_needed(local_label_zip, label_extract)

# Re-scan selected dataset root and its nested folders, including jpg/jpeg/png.
image_paths = find_images(DATA_DIR)
csv_paths = find_label_csvs(DATA_DIR)
print("Dataset directory:", DATA_DIR.resolve())
print("Local images found:", len(image_paths))
print("Local label CSVs:", csv_paths)
assert image_paths and csv_paths, (
    "Could not find local ISIC Task 3 images and ground-truth CSV. "
    "Set DATA_DIR to the folder containing the extracted dataset or ZIP files."
)
# Usually there is one Task 3 ground-truth CSV.
label_csv = next((p for p in csv_paths if "GroundTruth" in Path(p).name or "ground" in Path(p).name.lower()), csv_paths[0])
labels_df = pd.read_csv(label_csv)
print("Label CSV:", label_csv)
print(labels_df.head())
print("Columns:", labels_df.columns.tolist())

# Map image filename stem -> path
image_map = {Path(p).stem: p for p in image_paths}

# Expected first column is image ID; remaining columns are diagnosis one-hot labels.
id_col = labels_df.columns[0]
class_cols = list(labels_df.columns[1:])
assert len(class_cols) == 7, f"Expected 7 class columns, found {len(class_cols)}: {class_cols}"

rows = []
missing = 0
for _, row in labels_df.iterrows():
    image_id = str(row[id_col])
    path = image_map.get(image_id)
    if path is None:
        # Handle possible .jpg suffix in CSV ID
        path = image_map.get(Path(image_id).stem)
    if path is None:
        missing += 1
        continue
    label = int(np.argmax(row[class_cols].to_numpy(dtype=float)))
    rows.append({"image_id": Path(path).stem, "path": path, "label": label})

data_df = pd.DataFrame(rows).drop_duplicates("image_id").reset_index(drop=True)
print("Matched:", len(data_df), "Missing:", missing)
print("Ground-truth rows:", len(labels_df))
if missing:
    raise RuntimeError(
        f"Dataset is incomplete: {missing} labeled images were not found under {DATA_DIR}. "
        "Both classical and quantum scripts must use the same complete image folder. "
        "Set ISIC_DATA_DIR to the folder containing the full ISIC2018_Task3_Training_Input "
        "and GroundTruth files."
    )
if len(data_df) != len(labels_df):
    raise RuntimeError(f"Matched {len(data_df)} images but label CSV has {len(labels_df)} rows.")
print("Class counts:")
print(data_df["label"].value_counts().sort_index())
assert data_df["label"].nunique() == 7, "Expected all seven classes in matched data."
class_names = class_cols[:]  # labels follow the CSV's one-hot column order
# (i) shuffle all seven diagnosis classes; (ii) omit the final shuffled class.
class_ids = list(range(7))
rng = np.random.RandomState(SEED)
rng.shuffle(class_ids)

omitted_class_id = class_ids[-1]
selected_class_ids = class_ids[:-1]
# (iii) make three chronological tasks, introducing two classes per task (Task 1/2/3).
task_class_ids = [selected_class_ids[0:2], selected_class_ids[2:4], selected_class_ids[4:6]]

print("Seed:", SEED)
print("Shuffled class order:", [(i, class_names[i]) for i in class_ids])
print("Omitted class:", omitted_class_id, class_names[omitted_class_id])
print("Task 1:", [(i, class_names[i]) for i in task_class_ids[0]])
print("Task 2:", [(i, class_names[i]) for i in task_class_ids[1]])
print("Task 3:", [(i, class_names[i]) for i in task_class_ids[2]])

# Keep only six selected classes and remap them to contiguous labels 0..5 in task order.
old_to_new = {old: new for new, old in enumerate(selected_class_ids)}
selected_df = data_df[data_df.label.isin(selected_class_ids)].copy()
selected_df["global_label"] = selected_df["label"].map(old_to_new)

# Preserve original class IDs and names for reporting
new_class_names = [class_names[i] for i in selected_class_ids]
task_new_class_ids = [[old_to_new[c] for c in group] for group in task_class_ids]
# COURSEWORK REQUIREMENT 2: stratified 80/20 train/test split using the same seed.
train_df, test_df = train_test_split(
    selected_df,
    test_size=0.20,
    random_state=SEED,
    stratify=selected_df["global_label"],
    shuffle=True
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Train:", len(train_df), "Test:", len(test_df))
print("Train class counts:", train_df.global_label.value_counts().sort_index().to_dict())
print("Test class counts:", test_df.global_label.value_counts().sort_index().to_dict())
IMAGE_SIZE = 224
BATCH_SIZE = 8  # Lower VRAM use for 4 GB GPUs; increase only if memory allows
NUM_WORKERS = 0  # Windows-safe; avoids DataLoader multiprocessing spawn issues
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ISICClassificationDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(row.path).convert("RGB")
        image = self.transform(image)
        return image, int(row.global_label)

train_dataset = ISICClassificationDataset(train_df, train_transform)
test_dataset = ISICClassificationDataset(test_df, test_transform)

# Task-specific index lists in the training set; labels remain global 0..5.
task_train_indices = [
    train_df.index[train_df.global_label.isin(cls)].tolist()
    for cls in task_new_class_ids
]
task_test_frames = [
    test_df[test_df.global_label.isin(cls)].reset_index(drop=True)
    for cls in task_new_class_ids
]
print([len(x) for x in task_train_indices], [len(x) for x in task_test_frames])
def make_backbone():
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)
    net.fc = nn.Identity()
    # Freeze early layers; fine-tune layer4 and the head.
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("layer4")
    return net

# COURSEWORK REQUIREMENT 4: this compact classical head is parameter-count matched
# approximately to the quantum head; print the counts below and report the difference.
class ClassicalHead(nn.Module):
    # 512 -> 4 -> 6: 2,082 trainable parameters (including biases).
    def __init__(self, feature_dim=512, hidden_dim=4, n_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class ClassicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = make_backbone()
        self.head = ClassicalHead()
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

N_QUBITS = 4
N_Q_LAYERS = 2

def make_quantum_layer(n_qubits=N_QUBITS, n_layers=N_Q_LAYERS, shots=None):
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    grad_method = "backprop" if shots is None else "parameter-shift"

    @qml.qnode(dev, interface="torch", diff_method=grad_method)
    def circuit(inputs, weights):
        # Encode four reduced CNN features as rotation angles on 4 qubits.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        # Trainable variational ansatz with entangling gates.
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # Measure all computational-basis probabilities (2^4 = 16 outcomes).
        return qml.probs(wires=range(n_qubits))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)

# COURSEWORK REQUIREMENT 1: Hybrid quantum-classical head.
# Encoding: reduce 512 CNN features to 4 values and map them to Y-rotation angles.
# Ansatz: 2 layers of trainable StronglyEntanglingLayers on 4 qubits (<= 8 qubits).
# Measurement: return all 2^4=16 computational-basis probabilities.\n# Simulator: PennyLane default.qubit runs on CPU; CNN and reducer may run on CUDA.
# Class readout: sum probabilities whose basis index modulo 6 equals class 0..5,
# producing six nonnegative class scores; log(scores) are the logits.
class QuantumHead(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_Q_LAYERS, shots=None):
        super().__init__()
        self.reducer = nn.Linear(512, n_qubits)
        self.shots = shots
        self.quantum = make_quantum_layer(n_qubits, n_layers, shots=shots)
        # 16 basis probabilities are grouped into six class scores by summing
        # probabilities for basis indices whose index modulo 6 equals class id.
        self.n_qubits = n_qubits
        self.n_classes = 6
    def forward(self, features):
        # PennyLane's default.qubit simulator is CPU-based. Keep its inputs and
        # TorchLayer parameters on CPU, while allowing the CNN/reducer to use CUDA.
        output_device = features.device
        angles = torch.tanh(self.reducer(features)) * np.pi
        angles_cpu = angles.to("cpu")

        if self.shots is None:
            # Exact simulator supports batched inputs, so keep this fast path.
            probs = self.quantum(angles_cpu)
        else:
            # PennyLane parameter-shift currently does not support gradients
            # through broadcasted/batched tapes with trainable broadcasted
            # parameters. For finite-shot training, execute each sample as an
            # independent QNode call while sharing the same trainable weights.
            # Each call receives a 1-D angle vector, avoiding tape broadcasting.
            probs = torch.stack(
                [self.quantum(sample_angles) for sample_angles in angles_cpu],
                dim=0,
            )

        scores = []
        for c in range(self.n_classes):
            idx = [b for b in range(2 ** self.n_qubits) if b % self.n_classes == c]
            scores.append(probs[:, idx].sum(dim=1))
        scores = torch.stack(scores, dim=1).to(output_device)
        return torch.log(scores + 1e-8)

class QuantumModel(nn.Module):
    def __init__(self, shots=None):
        super().__init__()
        self.backbone = make_backbone()
        self.head = QuantumHead(shots=shots)

    def to(self, *args, **kwargs):
        # Move the CNN and classical reducer to the requested device, but keep
        # the PennyLane default.qubit TorchLayer on CPU to avoid CPU/CUDA mixing
        # inside the simulator. Mixed-device parameters are supported by AdamW.
        result = super().to(*args, **kwargs)
        self.head.quantum.to(torch.device("cpu"))
        return result

    def forward(self, x):
        return self.head(self.backbone(x))

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# COURSEWORK REQUIREMENT 4: report the trainable parameter-count match for the heads.
# Compare head-only parameters
c_head = ClassicalHead()
q_head = QuantumHead()
print("Classical head trainable params:", count_trainable(c_head))
print("Quantum head trainable params:", count_trainable(q_head))
print("Quantum simulator: PennyLane default.qubit on CPU; CNN uses", DEVICE)
EPOCHS_PER_TASK = 5  # Final coursework experiment setting
LR_BACKBONE = 1e-5
LR_HEAD = 1e-3

def make_loader(dataset, indices, shuffle):
    return DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

def build_test_loader(frame):
    ds = ISICClassificationDataset(frame, test_transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

test_loaders = [build_test_loader(f) for f in task_test_frames]

def train_one_task(model, loader, epochs=EPOCHS_PER_TASK):
    model.train()
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss_sum, seen = 0.0, 0
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
        print(f"  epoch {epoch+1}/{epochs} loss={loss_sum/max(seen,1):.4f}")

@torch.no_grad()
def evaluate_balanced(model, loader, task_class_ids):
    """Task-wise balanced accuracy: macro recall over only this task's true classes.

    Predictions of classes outside the task are counted as incorrect for the
    true class, but are not treated as additional classes in the macro average.
    This avoids sklearn's warning about predicted labels absent from y_true.
    """
    model.eval()
    correct_by_class = {int(c): 0 for c in task_class_ids}
    total_by_class = {int(c): 0 for c in task_class_ids}

    for images, labels in loader:
        images = images.to(DEVICE)
        predictions = model(images).argmax(dim=1).cpu().numpy()
        labels = labels.numpy()

        for true_label, predicted_label in zip(labels, predictions):
            true_label = int(true_label)
            predicted_label = int(predicted_label)
            if true_label in total_by_class:
                total_by_class[true_label] += 1
                if predicted_label == true_label:
                    correct_by_class[true_label] += 1

    recalls = [
        correct_by_class[c] / total_by_class[c]
        if total_by_class[c] else 0.0
        for c in task_class_ids
    ]
    return float(np.mean(recalls)) if recalls else float("nan")

# COURSEWORK REQUIREMENTS 3/4: stratified rehearsal memory from previous tasks.
# ratio is 5%, 10%, 20%, or 100% of each previously observed class's training examples.
def select_replay_indices(previous_indices, ratio, seed):
    if ratio <= 0 or not previous_indices:
        return []
    # Select replay examples stratified by class from previously observed training data.
    frame = train_df.loc[previous_indices]
    selected = []
    rng_local = np.random.RandomState(seed)
    for label, group in frame.groupby("global_label"):
        ids = group.index.to_numpy()
        n = max(1, int(round(len(ids) * ratio))) if ratio > 0 else 0
        n = min(n, len(ids))
        selected.extend(rng_local.choice(ids, size=n, replace=False).tolist())
    return sorted(set(selected))
# COURSEWORK REQUIREMENT 1/2: single shared six-class head; Naïve strategy is replay=0%.
# R[i,j] is balanced accuracy on test Task j after training through Task i.
# Average accuracy is the mean of final-row R; average forgetting is computed for Tasks 1-2.
def run_lifelong(model_kind="classical", rehearsal_ratio=0.0, shots=None, seed=SEED,
                 epochs=EPOCHS_PER_TASK, save_checkpoint=False):
    set_seed(seed)
    if model_kind == "classical":
        model = ClassicalModel().to(DEVICE)
    elif model_kind == "quantum":
        model = QuantumModel(shots=shots).to(DEVICE)
    else:
        raise ValueError("model_kind must be classical or quantum")

    R = np.full((3, 3), np.nan, dtype=float)
    previous_indices = []
    drift_by_task = []
    previous_head_state = None

    for task_idx, new_indices in enumerate(task_train_indices):
        replay_indices = select_replay_indices(previous_indices, rehearsal_ratio, seed + task_idx)
        combined_indices = sorted(set(new_indices + replay_indices))
        loader = make_loader(train_dataset, combined_indices, shuffle=True)

        # Record head parameter drift from the previous task checkpoint.
        if previous_head_state is not None:
            before = previous_head_state
        else:
            before = None

        print(f"\n{model_kind} | replay={rehearsal_ratio:.0%} | task {task_idx+1}: "
              f"new={len(new_indices)} replay={len(replay_indices)}")
        train_one_task(model, loader, epochs=epochs)

        current_state = {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()}
        if before is None:
            drift = 0.0
        else:
            sq = 0.0
            for k, v in current_state.items():
                if k in before and v.dtype.is_floating_point:
                    sq += torch.sum((v - before[k]) ** 2).item()
            drift = float(np.sqrt(sq))
        drift_by_task.append(drift)
        previous_head_state = current_state

        # Evaluate on each task test split seen so far, and future tasks too for a full matrix.
        for test_task_idx, loader_test in enumerate(test_loaders):
            R[task_idx, test_task_idx] = evaluate_balanced(
                model, loader_test, task_new_class_ids[test_task_idx]
            )

        previous_indices = sorted(set(previous_indices).union(new_indices))

    avg_accuracy = float(np.nanmean(R[2, :]))
    forgetting_per_task = []
    for j in range(2):  # tasks 1 and 2 can be forgotten after later learning
        best_after_learning = np.nanmax(R[j:, j])
        final_accuracy = R[2, j]
        forgetting_per_task.append(float(best_after_learning - final_accuracy))
    avg_forgetting = float(np.mean(forgetting_per_task)) if forgetting_per_task else 0.0

    result = {
        "model_kind": model_kind,
        "rehearsal_ratio": rehearsal_ratio,
        "shots": shots,
        "R": R,
        "average_accuracy": avg_accuracy,
        "average_forgetting": avg_forgetting,
        "head_parameter_drift": drift_by_task,
        "model": model,
    }
    if save_checkpoint:
        torch.save(model.state_dict(), f"{model_kind}_replay{int(rehearsal_ratio*100)}_seed{seed}.pt")
    return result

def print_result(result):
    print("R matrix (rows: after task i; columns: test task j):")
    print(pd.DataFrame(result["R"], index=["After T1","After T2","After T3"],
                      columns=["Test T1","Test T2","Test T3"]).round(4))
    print("Average final balanced accuracy:", round(result["average_accuracy"], 4))
    print("Average forgetting:", round(result["average_forgetting"], 4))
    print("Head parameter drift by task:", result["head_parameter_drift"])
# COURSEWORK REQUIREMENTS 3/4: classical runs for Naïve (0%), 5%, 10%, 20%, 100%.
# The same ratios are then evaluated for the hybrid quantum-classical model.
# Classical naïve and rehearsal runs


MODEL_KIND = "quantum"

class Tee:
    """Write terminal output to both the console and a proof log."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()
    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def save_run_proof(result, proof_dir, run_name, model_kind, rehearsal_ratio, shots, epochs):
    run_dir = proof_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.DataFrame(
        result["R"],
        index=["After_T1", "After_T2", "After_T3"],
        columns=["Test_T1", "Test_T2", "Test_T3"]
    )
    matrix.to_csv(run_dir / "accuracy_matrix.csv", index_label="training_stage")

    metrics = {
        "model_kind": model_kind,
        "rehearsal_ratio": rehearsal_ratio,
        "shots": shots,
        "seed": SEED,
        "epochs_per_task": epochs,
        "average_final_balanced_accuracy": result["average_accuracy"],
        "average_forgetting": result["average_forgetting"],
        "head_parameter_drift_by_task": result["head_parameter_drift"],
        "accuracy_matrix_R": np.asarray(result["R"]).tolist(),
        "task_class_names": [[new_class_names[c] for c in task] for task in task_new_class_ids],
        "omitted_class": class_names[omitted_class_id],
        "device": str(DEVICE),
        "train_images": len(train_df),
        "test_images": len(test_df),
        "head_trainable_parameters": count_trainable(result["model"].head),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "task_stage": ["After T1", "After T2", "After T3"],
        "head_parameter_drift": result["head_parameter_drift"]
    }).to_csv(run_dir / "parameter_drift.csv", index=False)

    # Save trained weights as evidence and for later inspection.
    torch.save(result["model"].state_dict(), run_dir / "model_state_dict.pt")
    return metrics


def main():
    print("\nStarting experiment main()...", flush=True)
    model_kind = MODEL_KIND
    ratios = [0.0, 0.05, 0.10, 0.20, 1.0]
    epochs = EPOCHS_PER_TASK

    proof_dir = Path(__file__).resolve().parent / "results" / f"{model_kind}_seed_{SEED}"
    proof_dir.mkdir(parents=True, exist_ok=True)
    log_path = proof_dir / "training_console.log"

    # Save the exact script used in the results folder.
    shutil.copy2(__file__, proof_dir / Path(__file__).name)

    all_metrics = []
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = Tee(original_out, log_file)
        sys.stderr = Tee(original_err, log_file)
        try:
            print("=" * 72)
            print(f"FULL {model_kind.upper()} LIFELONG LEARNING EXPERIMENT")
            print(f"Seed: {SEED} | epochs/task: {epochs} | device: {DEVICE}")
            print(f"Proof folder: {proof_dir}")
            print(f"Task classes: {[[new_class_names[c] for c in task] for task in task_new_class_ids]}")
            print(f"Omitted class: {class_names[omitted_class_id]}")
            print("=" * 72)

            for ratio in ratios:
                run_name = f"rehearsal_{int(ratio * 100)}pct"
                print(f"\n\n######## RUN: {run_name} ########")
                result = run_lifelong(
                    model_kind=model_kind,
                    rehearsal_ratio=ratio,
                    shots=None,
                    seed=SEED,
                    epochs=epochs,
                    save_checkpoint=False
                )
                print_result(result)
                metrics = save_run_proof(
                    result, proof_dir, run_name, model_kind, ratio, None, epochs
                )
                all_metrics.append(metrics)
                del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Quantum coursework requirement: repeat the best exact-probability
            # rehearsal setting with finite 1,024-shot measurements.
            if model_kind == "quantum":
                best_exact = max(
                    all_metrics,
                    key=lambda item: item["average_final_balanced_accuracy"]
                )
                best_ratio = best_exact["rehearsal_ratio"]
                run_name = f"best_rehearsal_{int(best_ratio * 100)}pct_1024_shots"
                print("\n\n######## FINITE-SHOT RUN ########")
                print("Selected rehearsal ratio by highest exact-run final average balanced accuracy:",
                      f"{best_ratio:.0%}")
                result = run_lifelong(
                    model_kind="quantum",
                    rehearsal_ratio=best_ratio,
                    shots=1024,
                    seed=SEED,
                    epochs=epochs,
                    save_checkpoint=False
                )
                print_result(result)
                metrics = save_run_proof(
                    result, proof_dir, run_name, model_kind, best_ratio, 1024, epochs
                )
                all_metrics.append(metrics)
                del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            summary = pd.DataFrame([{
                "model": m["model_kind"],
                "rehearsal_percent": int(m["rehearsal_ratio"] * 100),
                "shots": m["shots"] if m["shots"] is not None else "exact",
                "average_final_balanced_accuracy": m["average_final_balanced_accuracy"],
                "average_forgetting": m["average_forgetting"],
                "epochs_per_task": m["epochs_per_task"],
                "seed": m["seed"],
            } for m in all_metrics])
            summary.to_csv(proof_dir / "summary_results.csv", index=False)
            with open(proof_dir / "experiment_config.json", "w", encoding="utf-8") as f:
                json.dump({
                    "seed": SEED,
                    "epochs_per_task": epochs,
                    "model_kind": model_kind,
                    "rehearsal_percentages": [0, 5, 10, 20, 100],
                    "quantum_shots_run": 1024 if model_kind == "quantum" else None,
                    "dataset_path": str(DATA_DIR.resolve()),
                    "task_classes": [[new_class_names[c] for c in task] for task in task_new_class_ids],
                    "omitted_class": class_names[omitted_class_id],
                    "device": str(DEVICE),
                }, f, indent=2)

            print("\nAll runs completed.")
            print("Evidence saved to:", proof_dir)
            print(summary.to_string(index=False))
        finally:
            sys.stdout = original_out
            sys.stderr = original_err


if __name__ == "__main__":
    main()
