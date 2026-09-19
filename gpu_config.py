"""GPU 顯存配置。

TF 預設會在第一次用到 GPU 時預先佔掉幾乎全部顯存，和實際需要多少無關。
這台機器的 GPU 和別人共用，預設行為會把別人擠掉、或反過來讓自己起不來。
實測單一訓練工作在 batch 16 下只需要約 920 MiB
（CUDA context 388 + 模型與 optimizer 130 + activations 402），沒有理由佔住整張卡。

`set_memory_growth` 必須在 GPU 初始化之前呼叫，所以各進入點要在 import tensorflow
之後、任何運算之前呼叫 `enable_gpu_memory_growth()`。重複呼叫是安全的。
"""

from __future__ import annotations


def enable_gpu_memory_growth(verbose: bool = False) -> None:
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            if verbose:
                print(f"[gpu] {gpu.name} memory growth 已開啟")
        except RuntimeError as e:   # GPU 已經初始化就改不了了
            print(f"[warn] {gpu.name} 無法開啟 memory growth: {e}")
