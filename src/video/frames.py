from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CPUFrame:
    data: np.ndarray
    frame_id: int = 0

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def is_gpu(self) -> bool:
        return False

    def as_numpy(self) -> np.ndarray:
        return self.data

    def to_gpu(self, device: str = "cuda:0") -> "GPUFrame":
        tensor = torch.from_numpy(self.data).to(device)
        return GPUFrame(tensor=tensor, frame_id=self.frame_id)

    def crop(self, bbox):
        x1, y1, x2, y2 = bbox
        return self.data[y1:y2, x1:x2]


@dataclass
class GPUFrame:
    tensor: torch.Tensor
    frame_id: int = 0

    @property
    def width(self) -> int:
        return self.tensor.shape[1]

    @property
    def height(self) -> int:
        return self.tensor.shape[0]

    @property
    def is_gpu(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return str(self.tensor.device)

    def as_numpy(self) -> np.ndarray:
        return self.tensor.detach().to("cpu").numpy()

    def crop(self) -> torch.Tensor:
        raise NotImplementedError("use gpu_crop() for GPU-resident slicing")

    def gpu_crop(self, bbox) -> torch.Tensor:
        x1, y1, x2, y2 = bbox
        return self.tensor[y1:y2, x1:x2]


Frame = CPUFrame | GPUFrame


def ensure_numpy(frame) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        return frame
    return frame.as_numpy()
