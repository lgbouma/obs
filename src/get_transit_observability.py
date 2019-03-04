"""
given ra/dec and observatory location, get observable transits within a window
of time
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
from glob import glob

from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroplan import (FixedTarget, Observer, EclipsingSystem,
                       PrimaryEclipseConstraint, is_event_observable,
                       AtNightConstraint, AltitudeConstraint,
                       LocalTimeConstraint)

import datetime as dt

def minimal_example():
    apo = Observer.at_site('APO', timezone='US/Mountain')
    target = FixedTarget.from_name("HD 209458")

    primary_eclipse_time = Time(2452826.628514, format='jd')
    orbital_period = 3.52474859 * u.day
    eclipse_duration = 0.1277 * u.day

    hd209458 = EclipsingSystem(primary_eclipse_time=primary_eclipse_time,
                               orbital_period=orbital_period,
                               duration=eclipse_duration, name='HD 209458 b')

    n_transits = 100  # This is the roughly number of transits per year

    obs_time = Time('2017-01-01 12:00')
    midtransit_times = hd209458.next_primary_eclipse_time(
        obs_time, n_eclipses=n_transits)

    import astropy.units as u
    min_local_time = dt.time(18, 0)  # 18:00 local time at APO (7pm)
    max_local_time = dt.time(8, 0)  # 08:00 local time at APO (5am)
    constraints = [AtNightConstraint.twilight_civil(),
                   AltitudeConstraint(min=30*u.deg),
                   LocalTimeConstraint(min=min_local_time, max=max_local_time)]

    # just at midtime
    b = is_event_observable(constraints, apo, target, times=midtransit_times)

    # completely observable transits
    observing_time = Time('2016-01-01 00:00')

    ing_egr = hd209458.next_primary_ingress_egress_time(observing_time,
                                                        n_eclipses=n_transits)

    ibe = is_event_observable(constraints, apo, target,
                              times_ingress_egress=ing_egr)

    oot_duration = 30*u.minute
    oot_ing_egr = np.concatenate(
        (np.array(ing_egr[:,0] - oot_duration)[:,None],
         np.array(ing_egr[:,1] + oot_duration)[:,None]),
        axis=1)

    oibeo = is_event_observable(constraints, apo, target,
                                times_ingress_egress=oot_ing_egr)


def get_transit_observability(
    site, ra, dec, name, t_mid_0, period, duration, n_transits=100,
    obs_start_time=Time(dt.datetime.today().isoformat()),
    min_local_time=dt.time(16, 0),
    max_local_time=dt.time(9, 0),
    min_altitude=20*u.deg,
    oot_duration = 30*u.minute
):
    """
    note: barycentric corrections not yet implemented. (could do this myself!)
    -> 16 minutes of imprecision is baked into this observability calculator!

    args:

        site (astroplan.observer.Observer)

        ra, dec (units u.deg), e.g.:
            ra=101.28715533*u.deg, dec=16.71611586*u.deg,
        or can also accept
            ra="17 56 35.51", dec="-29 32 21.5"

        name (str), e.g., "Sirius"

        t_mid_0 (float): in BJD_TDB, preferably (but see note above).

        period (astropy quantity, units time)

        duration (astropy quantity, units time)

        n_transits (int): number of transits forward extrapolated to

        obs_start_time (astropy.Time object): when to start calculation from

        min_local_time, max_local_time: earliest time when you think observing
        is OK. E.g., 16:00 local and 09:00 local are earliest and latest. Note
        this constraint is a bit silly, since the astroplan "AtNightConstraint"
        is imposed automatically. As implemented, these are ignored.

        min_altitude (astropy quantity, units deg): 20 degrees is the more
        relevant constraint.

        oot_duration (astropy quantity, units time): with which to brack
        transit observations, to get an OOT baseline.
    """


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

    primary_eclipse_time = Time(t_mid_0, format='jd')

    system = EclipsingSystem(primary_eclipse_time=primary_eclipse_time,
                             orbital_period=period, duration=duration,
                             name=name)

    midtransit_times = system.next_primary_eclipse_time(
        obs_start_time, n_eclipses=n_transits)

    # for the time being, omit any local time constraints.
    constraints = [AtNightConstraint.twilight_civil(),
                   AltitudeConstraint(min=min_altitude)]
    #constraints = [AtNightConstraint.twilight_civil(),
    #               AltitudeConstraint(min=min_altitude),
    #               LocalTimeConstraint(min=min_local_time, max=max_local_time)]

    # observable just at midtime (bottom)
    b = is_event_observable(constraints, site, target, times=midtransit_times)

    # observable full transits (ingress, bottom, egress)
    ing_egr = system.next_primary_ingress_egress_time(obs_start_time,
                                                      n_eclipses=n_transits)

    ibe = is_event_observable(constraints, site, target,
                              times_ingress_egress=ing_egr)

    # completely observable transits (OOT, ingress, bottom, egress, OOT)
    oot_ing_egr = np.concatenate(
        (np.array(ing_egr[:,0] - oot_duration)[:,None],
         np.array(ing_egr[:,1] + oot_duration)[:,None]),
        axis=1)

    oibeo = is_event_observable(constraints, site, target,
                                times_ingress_egress=oot_ing_egr)

    ing_tmid_egr = np.concatenate(
        (np.array(ing_egr[:,0] - oot_duration)[:,None],
         np.array(midtransit_times)[:,None],
         np.array(ing_egr[:,1] + oot_duration)[:,None]),
        axis=1)

    return ibe, oibeo, ing_tmid_egr


def print_transit_observability(ibe, oibeo, ing_tmid_egr, site, ra, dec, name,
                                t_mid_0, period, duration, min_altitude,
                                oot_duration, outdir="../results/"):
    """
    write output to ../results/{name}_{site}.txt
    """

    # header information
    hdr = (
"""{} transit prediction

site = {}.
lon {:.2f}, lat {:.2f}, alt {:.1f}

RA = {}, DEC = {}.

NOTE: This calculator does not yet have barycenter correction implemented! This
means that all local times of astronomical events are at worst 16 minutes off.

Given:
        t0_BJD_TDB = {}
        period = {}
        duration = {}

Constrain to observe with:
        altitude > {}
        and
        night-time (nautical twilight).

Defined "out of transit" to be {} on either side of ingress/egress.

==========================================
OBSERVABLE IN OIBEO (ALL TIMES IN UTC)

epoch     t_ing                 t_mid                 t_egr
-------------------------------------------------------------------------------
""".format(
    name, site.name,
    site.location.lon, site.location.lat, site.location.height,
    ra, dec,
    t_mid_0, period, duration,
    min_altitude,
    oot_duration
)
    )

    n_oibeo = len(oibeo[oibeo])

    # ing_tmid_egr gives time of ingress/tmid/egress. To index with oibeo,
    # repeat it!
    oibeo_times = ing_tmid_egr[np.repeat(oibeo, 3, axis=0).T]

    lines = []
    for tra_ind in range(n_oibeo):

        # get ing/mid/egr times, formatted in iso, down to second precision.
        # format them to a width of 22 characters.
        t_ing_str = '{:s}'.format(
            oibeo_times[tra_ind*3].isot[:-4]).ljust(22,' ').replace('T',' ')
        t_mid_str = '{:s}'.format(
            oibeo_times[tra_ind*3+1].isot[:-4]).ljust(22,' ').replace('T',' ')
        t_egr_str = '{:s}'.format(
            oibeo_times[tra_ind*3+2].isot[:-4]).ljust(22,' ').replace('T',' ')

        t_mid_val = oibeo_times[tra_ind*3+1].jd
        epoch = int(np.round((t_mid_val - t_mid_0) / period.value, 0))

        # if you're not within 1.5 minutes = 1e-3 days of the mid-time being a
        # "true epoch", something is wrong with your calculation. raise error.
        if not (
            np.isclose(
                ((t_mid_val - t_mid_0)/period.value) % 1, 1, atol=1e-3
            ) or
            np.isclose(
                ((t_mid_val - t_mid_0)/period.value) % 0, 1, atol=1e-3
            )
        ):
            raise AssertionError('got error in tra_ind {}')

        epoch_str = '{}'.format(epoch).ljust(10,' ')

        this_line = '{}{}{}{}\n'.format(epoch_str,
                                      t_ing_str,
                                      t_mid_str,
                                      t_egr_str)

        lines.append(this_line)


    outname = '{}_{}.txt'.format(name, site.name.replace(' ','_'))
    outpath = os.path.join(outdir,outname)
    with open(outpath, mode='w') as f:
        f.writelines(hdr)
        f.writelines(lines)
    print('wrote {}'.format(outpath))


if __name__ == "__main__":


    # will change these a lot (todo: when in bulk -- write argparse interface)
    ##########################################
    #HIRES
    #site = Observer.at_site('W. M. Keck Observatory')
    #NEID
    #site = Observer.at_site('Kitt Peak National Observatory')
    #HARPS-N
    site = Observer.at_site('lapalma')

    ra="04 05 19.6"
    dec="+20 09 25.6"
    name = 'V1298_Tau_b'

    t_mid_0 = 2457091.18842 # BJD_TDB
    period = 24.13889*u.day
    duration = 6.386*u.hour

    # NOTE can omit
    # site = Observer.at_site('Las Campanas Observatory')
    # ra="17 56 35.51"
    # dec="-29 32 21.5"
    # name = 'OGLE-TR-56b'

    # t_mid_0 = 2453936.60070 # BJD_TDB
    # period = 1.21191096*u.day
    # duration = 0.0916*u.day
    # NOTE can omit

    # won't change these much
    ##########################################
    n_transits = 500
    obs_start_time = Time(dt.datetime.today().isoformat())
    min_altitude = 20*u.deg
    oot_duration = 30*u.minute

    ##########################################

    ibe, oibeo, ing_tmid_egr = get_transit_observability(
       site, ra, dec, name, t_mid_0, period, duration, n_transits=n_transits,
       obs_start_time=obs_start_time,
       min_altitude=min_altitude, oot_duration=oot_duration
    )

    outdir = "../results/{}".format(name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print_transit_observability(ibe, oibeo, ing_tmid_egr, site, ra, dec, name,
                                t_mid_0, period, duration, min_altitude,
                                oot_duration, outdir=outdir)
