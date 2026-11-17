"""ECG-only released-head predictor; device selection never changes the policy."""
from pathlib import Path
import os
import re
import numpy as np
from cardioagent.common.imports import upstream_module
from cardioagent.common.paths import code_root

def load_ecgfounder(root: Path):
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    source = code_root() / "third_party" / "ecgfounder"
    network = upstream_module("ecgfounder_net1d", source / "net1d.py")
    path = root / "1_lead_ECGFounder.pth"
    # The published checkpoint includes NumPy scalar training metadata.
    numpy_core = np._core if hasattr(np, "_core") else np.core
    numpy_types = [
        (numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype, np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType,
    ]
    with torch.serialization.safe_globals(numpy_types):
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    labels = (source / "tasks.txt").read_text(encoding="utf-8").splitlines()
    num_classes = state["dense.weight"].shape[0]
    if len(labels) != num_classes:
        raise ValueError("ECGFounder label count differs from the saved classification layer")
    model = network.Net1D(
        in_channels=1, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16,
        n_classes=num_classes, use_bn=False, use_do=False, return_features=True,
    )

    model.load_state_dict(state, strict=True)
    return model.eval(), labels, path

def normalize_ecg_device(value: str) -> str:
    """Accept CPU or exactly one CUDA device; importing this does not initialize CUDA."""
    if value == 'cpu':
        return value
    if value == 'cuda':
        return 'cuda:0'
    if isinstance(value, str) and re.fullmatch(r'cuda:[0-9]+', value):
        return f'cuda:{int(value.split(":")[1])}'
    raise ValueError('Device must be cpu or a single CUDA device, for example cuda:0')


def ecgfounder_execution(device: str = 'cpu') -> dict:
    """Fail explicitly when CUDA is requested but unavailable; never fall back to CPU."""
    import torch

    device = normalize_ecg_device(device)
    info = {'device': device, 'device_name': 'CPU', 'cuda_runtime': torch.version.cuda,
            'precision': 'float32', 'tf32': None, 'gpu_uuid': None,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    if device.startswith('cuda:'):
        index = int(device.split(':')[1])
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable; use --device cpu explicitly')
        if index >= torch.cuda.device_count():
            raise ValueError(f'{device} is outside the visible CUDA device range')
        properties = torch.cuda.get_device_properties(index)
        info.update(device_name=properties.name, gpu_uuid=str(getattr(properties, 'uuid', 'unknown')),
                    compute_capability=[properties.major, properties.minor], tf32=False,
                    cudnn_benchmark=False)
    return info


def create_ecgfounder_scorer(root: Path, device: str = 'cpu'):
    """Score an already preprocessed single-channel window using the released head."""
    import torch

    info = ecgfounder_execution(device)
    target = torch.device(info['device'])
    if target.type == 'cuda':
        torch.cuda.set_device(target)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    model, labels, path = load_ecgfounder(root)
    model = model.to(target).eval()
    if next(model.parameters()).device != target:
        raise RuntimeError('ECGFounder parameters are not on the requested device')
    if target.type == 'cuda':
        torch.cuda.synchronize(target)
    indices = [labels.index(name) for name in ("ATRIAL FIBRILLATION", "ATRIAL FLUTTER")]

    def predict(values: np.ndarray) -> tuple[float, float]:
        if values.shape != (5000,) or values.dtype != np.float32:
            raise ValueError("Expected a cached single-channel 500 Hz, 10-second float32 window")
        with torch.inference_mode():
            source = values if values.flags.writeable else values.copy()
            tensor = torch.from_numpy(source)[None, None, :].to(target)
            logits, _ = model(tensor)
            scores = logits[0, indices].sigmoid()
            # The blocking scalar return includes GPU execution and transfer in wall timing.
            result = scores.detach().cpu().tolist()
        return float(result[0]), float(result[1])

    return predict, {**info, "checkpoint": str(path.resolve()),
                     "scores": "uncalibrated_sigmoid", "input_samples": 5000}


# Compatibility with the original scorer name (cached means preprocessed input).
cached_ecgfounder_scorer = create_ecgfounder_scorer


def create_ecgfounder_label_scorer(root: Path, device: str = 'cpu'):
    """Offline 150-label export. The existing AF/AFL predictor stays unchanged."""
    import torch
    info=ecgfounder_execution(device)
    target=torch.device(info['device'])
    if target.type=='cuda':
        torch.cuda.set_device(target)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    model,labels,path=load_ecgfounder(Path(root))
    if len(labels)!=150 or len(set(labels))!=150:raise ValueError('Expected 150 unique ECGFounder labels')
    model=model.to(target).eval()
    if next(model.parameters()).device!=target:raise RuntimeError('Model device mismatch')
    if target.type=='cuda':torch.cuda.synchronize(target)
    def predict(values):
        if values.shape!=(5000,) or values.dtype!=np.float32 or not np.isfinite(values).all():
            raise ValueError('Expected finite single-channel 5000-point float32 model input')
        with torch.inference_mode():
            tensor=torch.from_numpy(values.copy())[None,None,:].to(target)
            logits,_=model(tensor)
            if logits.shape!=(1,150):raise ValueError('Unexpected classification-head dimensions')
            return logits[0].sigmoid().detach().cpu().numpy().copy()
    return predict,labels,{**info,'checkpoint':str(path.resolve()),'scores':'uncalibrated_sigmoid',
                          'input_samples':5000,'output_labels':len(labels)}
