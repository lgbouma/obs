import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
from glob import glob

from astropy.time import Time
from astropy.coordinates import get_body, get_sun, get_moon, SkyCoord
import astropy.units as u

from astroplan import (FixedTarget, Observer, EclipsingSystem,
                       PrimaryEclipseConstraint, is_event_observable,
                       AtNightConstraint, AltitudeConstraint,
                       LocalTimeConstraint, MoonSeparationConstraint,
                       moon)
from astroplan.plots import plot_airmass, dark_style_sheet

import datetime as dt

def main():

    site = Observer.at_site('Las Campanas Observatory')

    ra = "07:21:28.72"
    dec = "-45:34:03.85"
    name = "TIC_59859387.01"
    start_time = Time('2020-02-05 23:30:00')
    end_time = Time('2020-02-06 09:30:00')

    ##########################################

    range_time = end_time - start_time
    observe_time = start_time + range_time*np.linspace(0,1,100)

    if (isinstance(ra, u.quantity.Quantity) and
        isinstance(dec, u.quantity.Quantity)
    ):
        target_coord = SkyCoord(ra=ra, dec=dec)
    elif (isinstance(ra, str) and
          isinstance(dec, str)
    ):
        target_coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
    else:
        raise NotImplementedError

    target = FixedTarget(coord=target_coord, name=name)

    outdir = "../results/{}".format(name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outname = '{}_{}_start{}_airmass.png'.format(
        name, site.name.replace(' ','_'), repr(start_time.value)
    )
    outpath = os.path.join(outdir,outname)

    plot_airmass(target, site, observe_time, style_sheet=dark_style_sheet,
                 altitude_yaxis=True)

    plt.tight_layout()
    plt.savefig(outpath, dpi=250, bbox_inches='tight')
    print('made {}'.format(outpath))

if __name__ == "__main__":
    main()
