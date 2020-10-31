# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

from synpums.allocate import *  # noqa: F401 F403
from synpums.pums import *  # noqa: F401 F403
from synpums.acs import *  # noqa: F401 F403
from synpums.age_sex import age_sex_map # noqa: F401 F403
from synpums.marginals import *