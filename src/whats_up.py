'''
Make a csv file of [messier_id,ngc_id,v_mag,type,comments] V<11 objects that
are bright and up.

#FIXME
>>> python whats_up.py --help
usage: run_the_machine.py [-h] [-ir] [-N NSTARS] [-p] [-inj INJ] [-frd] [-c]
                          [-kicid KICID] [-nw NWORKERS] [-q]


Messier coordinates are retrieved from Simbad via astroquery.
NGC catalog (2000) was downloaded from HEASARC.
'''

# Generic imports
import numpy as np
import pandas as pd, matplotlib.pyplot as plt
import argparse

# Astropy imports
import astropy.units as u, astropy.constants as c
from astroquery.simbad import Simbad
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord

########################
# CATALOG CONSTRUCTION #
########################

def make_catalog(name):
    '''
    Prepares catalogs on hard drives (run one time).

    args:
        name (str): one in: ['messier', 'ngc_best']

    returns:
        nothing.
    '''
    if name == 'messier':
        data_path = '../data/demo_catalogs/messier_catalog.csv'
        cat = pd.read_csv(data_path, delimiter=';')

        out_cat = {}
        out_keys = ['messier_id', 'ngc_id', 'v_mag', 'type', 'comments']
        col_names = ['M', 'NGC', 'Mag', 'Type', 'Comments']
        for ix, out_key in enumerate(out_keys):
            out_cat[out_key] = np.array(cat[col_names[ix]])

        out_cat = pd.DataFrame(out_cat)
        out_cat = out_cat[out_keys] # fix column ordering

        # Get RA and DEC from SIMBAD via astroquery
        result_table = Simbad.query_object("m ???", wildcard=True)
        out_cat['ra'] = result_table['RA']
        out_cat['dec'] = result_table['DEC']
        out_cat['ref'] = result_table['COO_BIBCODE']
        out_cat['is_up'] = np.ones_like(out_cat['ra'])*np.nan

        # Get RA and Dec in decimal units. Issue: sometimes, have RA as (hr,
        # minute, second), sometimes as (hr, minute). Generally we only care up
        # to ~minute precision (this could also be solved with python-fu, but
        # why use it if the problem doesn't need it?). Same with DEC.
        ra = [float(ra.split(' ')[0])*u.hourangle +
              float(ra.split(' ')[1])/60.*u.hourangle
              for ra in out_cat['ra']]
        dec = [float(d.split(' ')[0])*u.degree +
               float(d[0]+'1')*float(d.split(' ')[1])*u.arcminute \
               for d in out_cat['dec']]

        out_cat['ra'] = np.around([r.value for r in ra], 3)
        out_cat['dec'] = np.around([d.value for d in dec], 3)
        out_cat = out_cat.sort_values('v_mag')
        out_cat.to_csv('../data/catalogs/messier.csv', index=False)


    elif name == 'ngc':
        data_path = '../data/demo_catalogs/NGC_full.txt'
        tab = ascii.read(data_path)
        tab = ascii.read(data_path,
                names=['nan','ngc_id','ra','dec','v_mag','type','comments','nan2'])

        # Drop wonky first row, and nan columns.
        tab = tab[1:]
        tab = tab['ngc_id','ra','dec','v_mag','type','comments']

        # Get RA and Dec in decimal units.
        ra = [float(ra.split(' ')[0])*u.hourangle +
              float(ra.split(' ')[1])/60.*u.hourangle for ra in tab['ra']]
        dec = [float(d.split(' ')[0])*u.degree +
               float(d[0]+'1')*float(d.split(' ')[1])*u.arcminute \
               for d in tab['dec']]

        # Convert to pandas and standard format (hackily).
        df = tab.to_pandas()
        df['ra'] = np.around([r.value for r in ra], 3)
        df['dec'] = np.around([d.value for d in dec], 3)
        df['ref'] = np.ones_like(df['ra'])*np.nan
        df['is_up'] = np.ones_like(df['ra'])*np.nan
        df['messier_id'] = np.ones_like(df['ra'])*np.nan

        out_keys = ['messier_id', 'ngc_id', 'v_mag', 'type', 'comments', 'ra',
                'dec', 'ref', 'is_up']
        df = df[out_keys] # order columns consistently

        # Save.
        df = df.sort_values('v_mag')
        df.to_csv('../data/catalogs/ngc.csv', index=False)


############################
# FIND WHAT'S UP FUNCTIONS #
############################

def compute_is_up(name, time):
    '''
    Computes what V<11 objects in a catalog are up right now.

    args:
        name (str): one in: ['messier', 'ngc']
        time (str): in 24hr format, e.g. "2017-04-17 21:00:00"

    returns:
        catalog with is_up calculated, sorted by v_mag.
    '''

    import datetime
    from astropy.time import Time
    from astroplan import FixedTarget
    from astropy.coordinates import SkyCoord
    from astroplan import Observer, FixedTarget, AltitudeConstraint, \
        AtNightConstraint, is_observable, is_always_observable, \
        months_observable

    assert name in ['messier', 'ngc']

    if name == 'messier':
        data_path = '../data/catalogs/messier.csv'
    elif name == 'ngc':
        data_path = '../data/catalogs/ngc.csv'

    df = pd.read_csv(data_path)
    # The search & target creation is slow for >~thousands of FixedTargets. NGC
    # catalog is 13226 objects. Take those with v_mag<11, since from Peyton we
    # likely won't go fainter.
    df = df.sort_values('v_mag')
    df = df[df['v_mag']<11]

    peyton = Observer(longitude=74.65139*u.deg, latitude=40.34661*u.deg,
        elevation=62*u.m, name='Peyton', timezone='US/Eastern')

    if name == 'ngc':
        ras = np.array(df['ra'])*u.hourangle
        decs = np.array(df['dec'])*u.degree
        names = np.array(df['ngc_id'])
    elif name == 'messier':
        ras = np.array(df['ra'])*u.hourangle
        decs = np.array(df['dec'])*u.degree
        names = np.array(df['messier_id'])

    targets = [FixedTarget(SkyCoord(ra=r, dec=d), name=n) for r, d, n in
            tuple(zip(ras, decs, names))]

    constraints = [AltitudeConstraint(10*u.deg, 82*u.deg),
            AtNightConstraint(max_solar_altitude=-3.*u.deg)]

    t_obs = Time(time)
    is_up = is_observable(constraints, peyton, targets, times=t_obs)

    df['is_up'] = is_up

    return df


def get_bright_objects_that_are_up(time):
    '''
    Wrapper function to get a big list of v_mag sorted stuff that is up.

    args:
        time (str): in 24hr format, e.g. "2017-04-17 21:00:00"

    returns:
        df (pd.DataFrame): v_mag objects that are up at the given time.
    '''

    print('getting which messier objects are up')
    messier = compute_is_up('messier', time)
    print('getting which ngc objects are up')
    ngc = compute_is_up('ngc', time)

    df = pd.concat([ngc, messier], ignore_index=True)
    df = df.sort_values('v_mag')

    # Find objects with both M and NGC IDs. Find their corresponding NGC
    # duplicates, and remove the duplicates (keeping the entries from
    # messier.csv, since it has better comments).
    objs = df[ ~(df['messier_id'].isnull() | df['ngc_id'].isnull()) ]
    for ngc_id in objs['ngc_id']:
        try:
            df = df.drop(df[df['ngc_id'] == 'NGC '+str(ngc_id)].index)
        except:
            continue

    out_df = df[df['is_up']==True].reset_index(drop=True)
    # messier_id column apparently cannot have int IDs and np.nans in the same
    # column. As a workaround, write it as a string.
    out_df['messier_id'] = list(map(lambda x: np.nan if np.isnan(x) \
            else str(int(x)), out_df['messier_id']))
    out_df = out_df[['messier_id','ngc_id','v_mag','type','comments']]

    return out_df


if __name__ == '__main__':

Make a csv file of [messier_id,ngc_id,v_mag,type,comments] V<11 objects that
are bright and up.

    parser = argparse.ArgumentParser(description=
            'Make a csv file of bright objects that are up at a given time')
    parser.add_argument('-irtest', '--injrecovtest', action='store_true',
        help='Inject and recover periodic transits for a small number of '+\
             'trial stars. Must specify N.')



    # RUN ONCE:
    #make_catalog('messier')
    #make_catalog('ngc')

    df = get_bright_objects_that_are_up('2017-04-17 22:00:00')
