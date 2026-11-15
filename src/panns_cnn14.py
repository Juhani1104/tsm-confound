"""PANNs Cnn14 without panns_inference writing to the home folder.

panns_inference's config module, run on import, downloads AudioSet's label list to
~/panns_data. Pointing HOME at PANNS_HOME for the import makes it use
PANNS_HOME/panns_data instead (ext/panns_data unless TSM_PANNS_HOME says otherwise);
HOME is restored right after. The model weights are loaded separately from paths.PANNS.
"""

import os

from paths import PANNS_HOME

_home = os.environ.get("HOME")
os.environ["HOME"] = str(PANNS_HOME)
try:
    from panns_inference.models import Cnn14  # noqa: F401
finally:
    if _home is None:
        del os.environ["HOME"]
    else:
        os.environ["HOME"] = _home
