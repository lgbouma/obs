'''
Example planning plot for observing the summer triangle.
'''
import numpy as np, astropy.units as u, matplotlib.pyplot as plt
from astroplan import Observer, FixedTarget
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astroplan.plots import plot_airmass

subaru = Observer.at_site('subaru')

altair = FixedTarget.from_name('Altair')
vega = FixedTarget.from_name('Vega')

coordinates = SkyCoord('20h41m25.9s', '+45d16m49.3s', frame='icrs')
deneb = FixedTarget(name='Deneb', coord=coordinates)

time = Time('2015-06-16 12:00:00')
sunset_tonight = subaru.sun_set_time(time, which='nearest')
sunrise_tonight = subaru.sun_rise_time(time, which='nearest')

# Are our targets up at the appropriate time?
subaru.target_is_up(time, altair)
subaru.target_is_up(time, vega)
subaru.is_night(time)

# What are optimal observation times, in terms of air mass?
altair_rise = subaru.target_rise_time(time, altair) + 5*u.minute
altair_set = subaru.target_set_time(time, altair) - 5*u.minute

plot_airmass(altair, subaru, time) 
plot_airmass(vega, subaru, time) 
plot_airmass(deneb, subaru, time)  
plt.legend(loc=1, bbox_to_anchor=(1, 1)) 
plt.savefig('stars_vs_airmass.png', dpi=200)


