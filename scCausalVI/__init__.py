from importlib.metadata import version
import warnings
from .model.scCausalVI import scCausalVIModel
warnings.filterwarnings("ignore")


package_name = 'scCausalVI'
try:
    __version__ = version("scCausalVI")
except ImportError:
    __version__ = "0.1.0"

__author__ = 'Shaokun An'

__all__ = [
    "scCausalVIModel"
]
