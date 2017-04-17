# Recommended workflow:
from astropy.Table import Table
target_table = Table.read('targets.txt', format='ascii')

# Create astroplan.FixedTarget objects for each one in the table.
from astropy.coordinates import SkyCoord
import astropy.units as u
targets = [(FixedTarget(coord=ra=ra*u.deg, dec=dec*u.deg), name=name)
            for name, ra, dec in target_table]

# Build a bulleted list of constrains:
# 1. Only observe btwn altitudes of 10-80 deg, with AltitudeConstraint class.
# 2. Put an upper limit on the airmass of each target with AirmassConstraint
# class.
# 3. Use the AtNightConstraint class too, to see things at night. We can define 
# night to be "between civil twilights" with the class method twilight_civil,
# but there are also other ways to define the observing window.

from astroplan import (AltitudeConstraint, AirmassConstraint,
        AtNightConstraint)
constraints = [AltitudeConstraint(10*u.deg, 80*u.deg), AirmassConstraint(5),
        AtNightConstraint.twilight_civil()]

from astroplan import is_observable, is_always_observable, months_observable
# Are targets *ever* observable in the time range?
ever_observable = is_observable(constraints, subaru, targets,
        time_range=time_range)
# Are targets *always* observable in the time range?
always_observable = is_always_observable(constraints, subaru, targets,
        time_range=time_range)
# During what months are the targets ever observable?
best_months = months_observable(constraints, subaru, targets)


