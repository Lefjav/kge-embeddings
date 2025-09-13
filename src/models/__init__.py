from .transe import TransE
from .rotate import RotatE
from .complex import ComplEx
from .conve import ConvE
from .convtranse import ConvTransE

def build_model(name, **kw):
    name = name.lower()
    if name == 'transe': return TransE(**kw)
    if name == 'rotate': return RotatE(**kw)
    if name == 'complex': return ComplEx(**kw)
    if name == 'conve': return ConvE(**kw)
    if name == 'convtranse': return ConvTransE(**kw)
    raise ValueError(f'Unknown model {name}')
