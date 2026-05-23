"""System resource monitor: CPU + RAM via psutil, NVIDIA GPU/VRAM via NVML."""
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class SysStats:
    cpu_pct: float
    ram_pct: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_pct: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    vram_pct: Optional[float] = None
    gpu_name: Optional[str] = None


class SystemMonitor:
    """Polls CPU/RAM (always) and NVIDIA GPU/VRAM (when available)."""

    def __init__(self) -> None:
        self._nvml = None
        self._handle = None
        self._gpu_name: Optional[str] = None
        self._try_init_nvml()
        # Prime psutil so the first sample isn't 0.0
        psutil.cpu_percent(interval=None)

    def _try_init_nvml(self) -> None:
        try:
            import pynvml  # provided by nvidia-ml-py package
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() == 0:
                return
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            self._gpu_name = name
        except Exception:
            self._nvml = None
            self._handle = None

    @property
    def has_gpu(self) -> bool:
        return self._nvml is not None and self._handle is not None

    @property
    def gpu_name(self) -> Optional[str]:
        return self._gpu_name

    def sample(self) -> SysStats:
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        stats = SysStats(
            cpu_pct=cpu,
            ram_pct=vm.percent,
            ram_used_gb=vm.used / (1024 ** 3),
            ram_total_gb=vm.total / (1024 ** 3),
        )
        if self.has_gpu:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                stats.gpu_pct = float(util.gpu)
                stats.vram_used_gb = mem.used / (1024 ** 3)
                stats.vram_total_gb = mem.total / (1024 ** 3)
                stats.vram_pct = (mem.used / mem.total * 100.0) if mem.total else 0.0
                stats.gpu_name = self._gpu_name
            except Exception:
                pass
        return stats

    def shutdown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None
            self._handle = None
