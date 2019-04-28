import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
from glob import glob

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

from astroplan.plots import plot_airmass
from astroplan import (FixedTarget, Observer, months_observable,
                       AtNightConstraint, AltitudeConstraint)

import datetime as dt



def make_airmass_chart(name="WASP 4"):

    target = FixedTarget.from_name(name)
    observer = Observer.at_site('keck')

    constraints = [AltitudeConstraint(min=20*u.deg, max=85*u.deg),
                   AtNightConstraint.twilight_civil()]

    best_months = months_observable(constraints, observer, [target])

    # computed observability on "best_months" grid of 0.5 hr
    print('for {}, got best-months on 0.5 hour grid:'.format(name))
    print(best_months)
    print('where 1 = Jan, 2 = Feb, etc.')



if __name__ == "__main__":

    make_airmass_chart()
