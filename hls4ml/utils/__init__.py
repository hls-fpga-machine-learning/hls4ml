from hls4ml.utils.config import config_from_keras_model, config_from_onnx_model, config_from_pytorch_model  # noqa: F401
from hls4ml.utils.example_models import fetch_example_list, fetch_example_model  # noqa: F401
from hls4ml.utils.plot import plot_model  # noqa: F401

try:
    from hls4ml.utils.editor import edit_model_configuration  # noqa: F401
except Exception:
    pass
