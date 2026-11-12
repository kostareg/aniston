"""Will the partner accept, and will the league let it through? Logistic models on perceived quantities."""
import math

import numpy as np

from aniston.config import settings


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def partner_week_weights(base: np.ndarray, is_playoff: np.ndarray, p_playoffs: float) -> np.ndarray:
    """How a partner weighs each remaining week, given their playoff odds p. `base` is the neutral weighting
    (playoff weeks up-weighted). A bubble team needs wins now, so regular-season weeks gain up to acc_urgency
    (at p = 50%); playoff weeks only count as far as the team expects to reach them."""
    s = settings()
    bubble = 4.0 * p_playoffs * (1.0 - p_playoffs)
    return np.where(is_playoff, base * p_playoffs, base * (1.0 + s.acc_urgency * bubble))


def p_accept(their_seen_lineup_gain: float, their_pvor_gain: float, partner_has_traded: bool,
             partner_p_playoffs: float = 1.0) -> float:
    """`their_seen_lineup_gain` should already be weighted by `partner_week_weights`."""
    s = settings()
    z = s.acc_bias + s.acc_w_lineup * their_seen_lineup_gain + s.acc_w_value * their_pvor_gain
    if s.acc_dead_below > 0:  # all but eliminated: nothing to play for, so offers mostly go unanswered
        z -= s.acc_dead_penalty * max(0.0, 1.0 - partner_p_playoffs / s.acc_dead_below)
    return _sigmoid(z + (s.acc_active_bonus if partner_has_traded else 0.0))


def p_no_veto(my_pvor_gain: float) -> float:
    s = settings()
    return _sigmoid(s.veto_bias - s.veto_w * max(0.0, my_pvor_gain))
