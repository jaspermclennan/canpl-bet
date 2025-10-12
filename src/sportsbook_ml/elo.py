def elo_update(r_a, r_b, score_a, k=20):
    """Return new rating for team A after a match vs team B.
    score_a: 1=win, 0.5=draw, 0=loss
    """
    exp_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    return r_a + k * (score_a - exp_a)
