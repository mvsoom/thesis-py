# Formant bandwidth bounds for adult males and females
# ----------------------------------------------------
#
# We select these from following sources:
#
# [V]   @Vallee1994
# [K]   @Kent2018       Table 5
# [S]   @Stevens2000    Figure 6.1 p. 259
# [k]   @Klatt1980      Table I
#
# Note that "formant bandwidth data from LPC should be considered as tentative
# and potentially highly inaccurate" [@Kent2018, p. 88], so we take wide
# intervals based on intuitive judgment. In general, though, rounding is to
# nearest tenth. In the case of [k], the ranges given are "permitted ranges of
# values [for the formant synthesizer]", so need to be taken with a bit of
# salt.
#
# Bn    min     max     source
# ----- ------- ------- -------
# B1    30      110     V
#       10      300     K
#       20      130     S
#       40      500     k
#
# B2    10      70      V
#       40      400     K
#       20      200     S
#       40      500     k
#
# B3    10      160     V
#       60      230     K
#       20      200     S
#       40      500     k
#
# B4    20      230     V
#       20      250     S
#       100     500     k
#
# B5    20      350     V
#       50      300     S
#       150     700     k
#
# B6    200     2000    k
def bounds():
    return [(10., 20., 20., 20, 20.), (300., 300., 400., 400., 500.)]
