"""Will the partner accept, and will the league let it through? Logistic models on perceived quantities."""
import math

from aniston.config import settings


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def p_accept(their_seen_lineup_gain: float, their_pvor_gain: float, partner_has_traded: bool) -> float:
    s = settings()
    z = s.acc_bias + s.acc_w_lineup * their_seen_lineup_gain + s.acc_w_value * their_pvor_gain
    return _sigmoid(z + (s.acc_active_bonus if partner_has_traded else 0.0))


def p_no_veto(my_pvor_gain: float) -> float:
    s = settings()
    return _sigmoid(s.veto_bias - s.veto_w * max(0.0, my_pvor_gain))
