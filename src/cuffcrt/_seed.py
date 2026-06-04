"""Single source of truth for the global random seed.

Import this everywhere stochastic. Carried forward from the originating
analysis so regenerated results match. ``RNG`` is a module-level generator
seeded once at import; pass an explicit ``np.random.default_rng(GLOBAL_SEED)``
where bit-stable, isolated streams are required (for example, in tests).
"""

import numpy as np

GLOBAL_SEED = 20260426
RNG = np.random.default_rng(GLOBAL_SEED)
