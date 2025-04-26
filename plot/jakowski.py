from math import sin, cos, pi, sqrt, radians
from datetime import datetime as dt
from statistics import mean
import seaborn as sns

from aacgmv2 import get_aacgm_coord
from dateutil import tz
from timezonefinder import TimezoneFinder

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import norm

from plot.utils import (
    convert_iso_to_day_of_year,
    get_month_days_count,
    get_sunrise_sunset,
)
from plot.graph import ( plot_graph )
from dal.models import (
    select_coords_by_ursi,
    select_solar_flux_day_mean,
    select_solar_flux_81_mean,
    select_f0f2_sat_tec,
)
from dal.handlers import (
    get_gap_spread_for_month,
    get_gap_spread_for_sum_win,
    get_gap_spread_for_year,
)

c1 = 0.25031
c2 = -0.10451
c3 = 0.12037
c4 = -0.01268
c5 = -0.00779
c6 = 0.03171
c7 = -0.11763
c8 = 0.06199
c9 = -0.01147
c10 = 0.03417
c11 = 302.44989
c12 = 0.00474


def utc_to_local(date, time, lat, long):
    tf = TimezoneFinder()

    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz(tf.timezone_at(lng=long, lat=lat))

    utc = dt.strptime(' '.join([date, time]), '%Y-%m-%d %H:%M:%S')

    utc = utc.replace(tzinfo=from_zone)
    local_dt = utc.astimezone(to_zone)
    
    h = float(dt.strftime(local_dt, '%H'))

    return h + (int(dt.strftime(local_dt, '%M')) / 60)


# lat in radians
def calc_cos_hi(lat, delta):
    return sin(lat) * sin(delta) + cos(lat) * cos(delta)


def calc_cos_hi2(cos_hi, lat, delta):
    return cos_hi - (2 * lat / pi) * sin(delta)


def calc_cos_hi3(cos_hi, SD=0.8):
    return sqrt(cos_hi + SD)


# lat in degree
def F1(
    date: str,
    time: str,
    lat,
    long,
    LTd=2,
    LTsd=9,
    LTtd: int=4,
    LTqd: int=6,
):
    LT = utc_to_local(date, time, lat, long)
    
    lat = radians(lat)
    DVm = 2*pi*(LT - LTd)/24
    SDVm = 2*pi*(LT - LTsd)/12
    TDVm = 2*pi*(LT - LTtd)/8
    QDVm = 2*pi*(LT - LTqd)/6
    
    
    N = convert_iso_to_day_of_year(date)
    delta = radians(-23.44 * cos(radians(360/365 * (N + 10))))
    
    cos_hi = calc_cos_hi(lat, delta)
    cos_hi2 = calc_cos_hi2(cos_hi, lat, delta)
    cos_hi3 = calc_cos_hi3(cos_hi)
    
    return cos_hi3 + (
        c1 * cos(DVm) +
        c2 * cos(SDVm) +
        c3 * cos(TDVm) +
        c4 * cos(QDVm)
    ) * cos_hi2


def F2(date, doya=340, doysa=360):
    doy = convert_iso_to_day_of_year(date)
    AVm = 2 * pi * (doy - doya) / 365.25
    SAVm = 4 * pi * (doy - doysa) / 365.25
    
    return 1 + c5 * cos(AVm) + c6 * cos(SAVm)

# lat in degree
def F3(lat: float, long: float, date: str):
    try:
        geom_lat = radians(
            get_aacgm_coord(lat, long, 0, dt.strptime(date, '%Y-%m-%d'))[0]
        )
    except Exception as ex:
        print(ex)

    return (
        1 + c7*cos(2*pi*geom_lat/180) +
        c8*sin(2*pi*geom_lat/80) +
        c9*sin(2*pi*geom_lat/55) +
        c10*cos(2*pi*geom_lat/40)
    )


def calc_F10_1(F10, F10_81):
    return (0.8*F10 + 1.2*F10_81)/2


def calc_F10_2(F10_1):
    return (F10_1 - 90)**2


def F4(date: str):
    F10 = select_solar_flux_day_mean(date)
    F10_81 = select_solar_flux_81_mean(date)
    
    return c11 + c12*(calc_F10_2(calc_F10_1(F10, F10_81)))


def calc_tau(
    date: str,
    time: str,
    lat: float,
    long: float,
):
    if long > 180:
        long = long - 360
    return (
        F1(date, time, lat, long) *
        F2(date) *
        F3(lat, long, date) *
        F4(date)
    )


Z = 1/12.4 * 1e4

def calc_k(
    date: str,
    time: str,
    lat: float,
    long: float,
):
    return Z / calc_tau(date, time, lat, long)


def calc_f0F2(k, TEC):
    return sqrt(k * TEC)


def get_jmodel_k_spread_for_month(ursi, month, year):
    str_month = f'0{month}' if month < 10 else str(month)
    jk_sun = []
    jk_moon = []

    crd = select_coords_by_ursi(ursi)
    hour = [f'0{t}' if t < 10 else f'{t}' for t in range(23)]

    for d in range(1, get_month_days_count(month) + 1):
        day = f'0{d}' if d < 10 else str(d)
        date = f'{year}-{str_month}-{day}'

        data = []
        for h in hour:
            data.append((h, calc_k(date, f'{h}:00:00', crd['lat'], crd['long'])))

        sun, moon = split_to_sun_moon(data, ursi, date)
        jk_sun.append(mean([v[1] for v in sun]))
        jk_moon.append(mean([v[1] for v in moon]))


    return jk_sun, jk_moon


def split_to_sun_moon(data, ursi, date):
    sunrise, sunset = get_sunrise_sunset(date, select_coords_by_ursi(ursi))

    if sunrise < sunset:
        # sun = df[(hour >= sunrise) & (hour < sunset)]
        # moon = df[(hour < sunrise) | (hour >= sunset)]
        sun = [r for r in data if (r[0] >= sunrise) and (r[0] < sunset)]
        moon = [r for r in data if (r[0] < sunrise) or (r[0] >= sunset)]
    else:
        # sun = df[(hour >= sunrise) | (hour < sunset)]
        # moon = df[(hour < sunrise) & (hour >= sunset)]
        sun = [r for r in data if (r[0] >= sunrise) or (r[0] < sunset)]
        moon = [r for r in data if (r[0] < sunrise) and (r[0] >= sunset)]
        
    return sun, moon


def calc_f0f2_k_jmodel_gap_for_day(ursi, date):
    coords = select_coords_by_ursi(ursi)

    lat, long = coords['lat'], coords['long']
    row_data = select_f0f2_sat_tec(ursi, date)
    sun, moon = split_to_sun_moon(row_data, ursi, date)

    k_sun = [r[1]**2/r[2] for r in sun]
    k_moon = [r[1]**2/r[2] for r in moon]
    f0f2_sun = [r[1] for r in sun]
    f0f2_moon = [r[1] for r in moon]

    jmodel_k_sun = [calc_k(date, s[0]+':00:00', lat, long) for s in sun]
    jmodel_k_moon = [calc_k(date, m[0]+':00:00', lat, long) for m in moon]
    jmodel_f0f2_sun = [calc_f0F2(k, r[2]) for k, r in zip(jmodel_k_sun, sun)]
    jmodel_f0f2_moon = [calc_f0F2(k, r[2]) for k, r in zip(jmodel_k_moon, moon)]

    gap_f0f2_sun = round(mean([abs(f - j) for f, j in zip(f0f2_sun, jmodel_f0f2_sun)]), 2)
    gap_f0f2_moon = round(mean([abs(f - j) for f, j in zip(f0f2_moon, jmodel_f0f2_moon)]), 2)
    gap_k_sun = round(mean([abs(k - j) for k, j in zip(k_sun, jmodel_k_sun)]), 2)
    gap_k_moon = round(mean([abs(k - j) for k, j in zip(k_moon, jmodel_k_moon)]), 2)

    return {
        'gap_f0f2_sun': gap_f0f2_sun,
        'gap_f0f2_moon': gap_f0f2_moon,
        'gap_k_sun': gap_k_sun,
        'gap_k_moon': gap_k_moon,
        'jmodel_sun_k': mean(jmodel_k_sun),
        'jmodel_moon_k': mean(jmodel_k_moon),
    }


def plot_spread_gap_for_month(
    ursi: str,
    month: int,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    coords = select_coords_by_ursi(ursi)
    gap_f0f2_sun, gap_f0f2_moon, gap_k_sun, gap_k_moon = get_gap_spread_for_month(ursi, month, year)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year} Month: {month}",
        fontsize=18, y=0.99,
    )

    ax[0][0].set_title('gap_f0f2_sun', fontsize=15)
    ax[0][1].set_title('gap_f0f2_moon', fontsize=15)
    ax[1][0].set_title('gap_k_sun', fontsize=15)
    ax[1][1].set_title('gap_k_moon', fontsize=15)

    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    
    ax[0][0].set_xlim(x_lim[0], x_lim[1])
    ax[0][0].set_ylim(y_lim[0], y_lim[1])
    ax[0][1].set_xlim(x_lim[0], x_lim[1])
    ax[0][1].set_ylim(y_lim[0], y_lim[1])
    ax[1][0].set_xlim(x_lim[0], x_lim[1])
    ax[1][0].set_ylim(y_lim[0], y_lim[1])
    ax[1][1].set_xlim(x_lim[0], x_lim[1])
    ax[1][1].set_ylim(y_lim[0], y_lim[1])

    sns.histplot(gap_f0f2_sun, kde=True, ax=ax[0][0])
    sns.histplot(gap_f0f2_moon, kde=True, ax=ax[0][1])
    sns.histplot(gap_k_sun, kde=True, ax=ax[1][0])
    sns.histplot(gap_k_moon, kde=True, ax=ax[1][1])
    
    avr_gap_f0f2_sun, std_gap_f0f2_sun = norm.fit(gap_f0f2_sun)
    text_gap_f0f2_sun = '\n'.join((
    r'$k=%.2f$' % (avr_gap_f0f2_sun, ),
    r'$\sigma^2=%.2f$' % (std_gap_f0f2_sun, )))
    
    avr_gap_f0f2_moon, std_gap_f0f2_moon = norm.fit(gap_f0f2_moon)
    text_gap_f0f2_moon = '\n'.join((
    r'$k=%.2f$' % (avr_gap_f0f2_moon, ),
    r'$\sigma^2=%.2f$' % (std_gap_f0f2_moon, )))
    
    avr_gap_k_sun, std_gap_k_sun = norm.fit(gap_k_sun)
    text_gap_k_sun = '\n'.join((
    r'$k=%.2f$' % (avr_gap_k_sun, ),
    r'$\sigma^2=%.2f$' % (std_gap_k_sun, )))

    avr_gap_k_moon, std_gap_k_moon = norm.fit(gap_k_moon)
    text_gap_k_moon = '\n'.join((
    r'$k=%.2f$' % (avr_gap_k_moon, ),
    r'$\sigma^2=%.2f$' % (std_gap_k_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, text_gap_f0f2_sun, transform=ax[0][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, text_gap_f0f2_moon, transform=ax[0][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][0].text(0.05, 0.95, text_gap_k_sun, transform=ax[1][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, text_gap_k_moon, transform=ax[1][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


def plot_spread_gap_for_sum_win(
    ursi: str,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    coords = select_coords_by_ursi(ursi)
    spread = get_gap_spread_for_sum_win(ursi, year)
    sum_gap_f0f2_sun, sum_gap_f0f2_moon, sum_gap_k_sun, sum_gap_k_moon = spread['sum']
    win_gap_f0f2_sun, win_gap_f0f2_moon, win_gap_k_sun, win_gap_k_moon = spread['win']

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year}",
        fontsize=18, y=0.99,
    )

    ax[0][0].set_title('gap_f0f2_sun', fontsize=15)
    ax[0][1].set_title('gap_f0f2_moon', fontsize=15)
    ax[1][0].set_title('gap_k_sun', fontsize=15)
    ax[1][1].set_title('gap_k_moon', fontsize=15)
    
    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    
    ax[0][0].set_xlim(x_lim[0], x_lim[1])
    ax[0][0].set_ylim(y_lim[0], y_lim[1])
    ax[0][1].set_xlim(x_lim[0], x_lim[1])
    ax[0][1].set_ylim(y_lim[0], y_lim[1])
    ax[1][0].set_xlim(x_lim[0], x_lim[1])
    ax[1][0].set_ylim(y_lim[0], y_lim[1])
    ax[1][1].set_xlim(x_lim[0], x_lim[1])
    ax[1][1].set_ylim(y_lim[0], y_lim[1])

    sns.histplot(sum_gap_f0f2_sun, kde=True, ax=ax[0][0], color='red')
    sns.histplot(sum_gap_f0f2_moon, kde=True, ax=ax[0][1], color='red')
    sns.histplot(sum_gap_k_sun, kde=True, ax=ax[1][0], color='red')
    sns.histplot(sum_gap_k_moon, kde=True, ax=ax[1][1], color='red')

    sns.histplot(win_gap_f0f2_sun, kde=True, ax=ax[0][0])
    sns.histplot(win_gap_f0f2_moon, kde=True, ax=ax[0][1])
    sns.histplot(win_gap_k_sun, kde=True, ax=ax[1][0])
    sns.histplot(win_gap_k_moon, kde=True, ax=ax[1][1])

    avr_sum_gap_f0f2_sun, std_sum_gap_f0f2_sun = norm.fit(sum_gap_f0f2_sun)
    text_sum_gap_f0f2_sun = '\n'.join((
    r'$k=%.2f$' % (avr_sum_gap_f0f2_sun, ),
    r'$\sigma^2=%.2f$' % (std_sum_gap_f0f2_sun, )))

    avr_sum_gap_f0f2_moon, std_sum_gap_f0f2_moon = norm.fit(sum_gap_f0f2_moon)
    text_sum_gap_f0f2_moon = '\n'.join((
    r'$k=%.2f$' % (avr_sum_gap_f0f2_moon, ),
    r'$\sigma^2=%.2f$' % (std_sum_gap_f0f2_moon, )))

    avr_sum_gap_k_sun, std_sum_gap_k_sun = norm.fit(sum_gap_k_sun)
    text_sum_gap_k_sun = '\n'.join((
    r'$k=%.2f$' % (avr_sum_gap_k_sun, ),
    r'$\sigma^2=%.2f$' % (std_sum_gap_k_sun, )))

    avr_sum_gap_k_moon, std_sum_gap_k_moon = norm.fit(sum_gap_k_moon)
    text_sum_gap_k_moon = '\n'.join((
    r'$k=%.2f$' % (avr_sum_gap_k_moon, ),
    r'$\sigma^2=%.2f$' % (std_sum_gap_k_moon, )))

    avr_win_gap_f0f2_sun, std_win_gap_f0f2_sun = norm.fit(win_gap_f0f2_sun)
    text_win_gap_f0f2_sun = '\n'.join((
    r'$k=%.2f$' % (avr_win_gap_f0f2_sun, ),
    r'$\sigma^2=%.2f$' % (std_win_gap_f0f2_sun, )))
    
    avr_win_gap_f0f2_moon, std_win_gap_f0f2_moon = norm.fit(win_gap_f0f2_moon)
    text_win_gap_f0f2_moon = '\n'.join((
    r'$k=%.2f$' % (avr_win_gap_f0f2_moon, ),
    r'$\sigma^2=%.2f$' % (std_win_gap_f0f2_moon, )))
    
    avr_win_gap_k_sun, std_win_gap_k_sun = norm.fit(win_gap_k_sun)
    text_win_gap_k_sun = '\n'.join((
    r'$k=%.2f$' % (avr_win_gap_k_sun, ),
    r'$\sigma^2=%.2f$' % (std_win_gap_k_sun, )))

    avr_win_gap_k_moon, std_win_gap_k_moon = norm.fit(win_gap_k_moon)
    text_win_gap_k_moon = '\n'.join((
    r'$k=%.2f$' % (avr_win_gap_k_moon, ),
    r'$\sigma^2=%.2f$' % (std_win_gap_k_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, text_sum_gap_f0f2_sun, transform=ax[0][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, text_sum_gap_f0f2_moon, transform=ax[0][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][0].text(0.05, 0.95, text_sum_gap_k_sun, transform=ax[1][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, text_sum_gap_k_moon, transform=ax[1][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    ax[0][0].text(0.95, 0.95, text_win_gap_f0f2_sun, transform=ax[0][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[0][1].text(0.95, 0.95, text_win_gap_f0f2_moon, transform=ax[0][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][0].text(0.95, 0.95, text_win_gap_k_sun, transform=ax[1][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][1].text(0.95, 0.95, text_win_gap_k_moon, transform=ax[1][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


def plot_spread_gap_for_year(
    ursi: str,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    coords = select_coords_by_ursi(ursi)
    gap_f0f2_sun, gap_f0f2_moon, gap_k_sun, gap_k_moon = get_gap_spread_for_year(ursi, year)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year}",
        fontsize=18, y=0.99,
    )

    ax[0][0].set_title('gap_f0f2_sun', fontsize=15)
    ax[0][1].set_title('gap_f0f2_moon', fontsize=15)
    ax[1][0].set_title('gap_k_sun', fontsize=15)
    ax[1][1].set_title('gap_k_moon', fontsize=15)

    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    
    ax[0][0].set_xlim(x_lim[0], x_lim[1])
    ax[0][0].set_ylim(y_lim[0], y_lim[1])
    ax[0][1].set_xlim(x_lim[0], x_lim[1])
    ax[0][1].set_ylim(y_lim[0], y_lim[1])
    ax[1][0].set_xlim(x_lim[0], x_lim[1])
    ax[1][0].set_ylim(y_lim[0], y_lim[1])
    ax[1][1].set_xlim(x_lim[0], x_lim[1])
    ax[1][1].set_ylim(y_lim[0], y_lim[1])

    sns.histplot(gap_f0f2_sun, kde=True, ax=ax[0][0])
    sns.histplot(gap_f0f2_moon, kde=True, ax=ax[0][1])
    sns.histplot(gap_k_sun, kde=True, ax=ax[1][0])
    sns.histplot(gap_k_moon, kde=True, ax=ax[1][1])
    
    avr_gap_f0f2_sun, std_gap_f0f2_sun = norm.fit(gap_f0f2_sun)
    text_gap_f0f2_sun = '\n'.join((
    r'$k=%.2f$' % (avr_gap_f0f2_sun, ),
    r'$\sigma^2=%.2f$' % (std_gap_f0f2_sun, )))
    
    avr_gap_f0f2_moon, std_gap_f0f2_moon = norm.fit(gap_f0f2_moon)
    text_gap_f0f2_moon = '\n'.join((
    r'$k=%.2f$' % (avr_gap_f0f2_moon, ),
    r'$\sigma^2=%.2f$' % (std_gap_f0f2_moon, )))
    
    avr_gap_k_sun, std_gap_k_sun = norm.fit(gap_k_sun)
    text_gap_k_sun = '\n'.join((
    r'$k=%.2f$' % (avr_gap_k_sun, ),
    r'$\sigma^2=%.2f$' % (std_gap_k_sun, )))

    avr_gap_k_moon, std_gap_k_moon = norm.fit(gap_k_moon)
    text_gap_k_moon = '\n'.join((
    r'$k=%.2f$' % (avr_gap_k_moon, ),
    r'$\sigma^2=%.2f$' % (std_gap_k_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, text_gap_f0f2_sun, transform=ax[0][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, text_gap_f0f2_moon, transform=ax[0][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][0].text(0.05, 0.95, text_gap_k_sun, transform=ax[1][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, text_gap_f0f2_moon, transform=ax[1][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


def plot_compare_jmodel_ion_f0f2(ursi, date):
    coords = select_coords_by_ursi(ursi)
    sunrise, sunset = get_sunrise_sunset(date, select_coords_by_ursi(ursi))
    lat, long = coords['lat'], coords['long']

    row_data = select_f0f2_sat_tec(ursi, date)
    sun, moon = split_to_sun_moon(row_data, ursi, date)
    hour_sun = [s[0] for s in sun]
    hour_moon = [s[0] for s in moon]
    # hour = [r[0] for r in row_data]
    # f0f2 = [r[1] for r in row_data]
    # k = [r[1]**2/r[2] for r in row_data]
    f0f2_sun = [r[1] for r in sun]
    f0f2_moon = [r[1] for r in moon]

    k_sun = [r[1]**2/r[2] for r in sun]
    k_moon = [r[1]**2/r[2] for r in moon]
    jmodel_k_sun = [calc_k(date, s[0]+':00:00', lat, long) for s in sun]
    jmodel_k_moon = [calc_k(date, m[0]+':00:00', lat, long) for m in moon]

    jmodel_f0f2_sun = [calc_f0F2(k, r[2]) for k, r in zip(jmodel_k_sun, sun)]
    jmodel_f0f2_moon = [calc_f0F2(k, r[2]) for k, r in zip(jmodel_k_moon, moon)]

    gap_f0f2_sun = round(mean([abs(f - j) for f, j in zip(f0f2_sun, jmodel_f0f2_sun)]), 2)
    gap_f0f2_moon = round(mean([abs(f - j) for f, j in zip(f0f2_moon, jmodel_f0f2_moon)]), 2)

    gap_k_sun = round(mean([abs(k - j) for k, j in zip(k_sun, jmodel_k_sun)]), 2)
    gap_k_moon = round(mean([abs(k - j) for k, j in zip(k_moon, jmodel_k_moon)]), 2)

    # jmodel_f0f2 = [
    #     calc_f0F2(calc_k(date, r[0]+':00:00', lat, long), r[2])
    #     for r in row_data
    # ]

    # jmodel_k = [calc_k(date, h+':00:00', lat, long) for h in hour]


    r1_sun = round(pearsonr(f0f2_sun, jmodel_f0f2_sun)[0], 2)
    r1_moon = round(pearsonr(f0f2_moon, jmodel_f0f2_moon)[0], 2)
    r2_sun = round(pearsonr(k_sun, jmodel_k_sun)[0], 2)
    r2_moon = round(pearsonr(k_moon, jmodel_k_sun)[0], 2)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,15))
    fig.suptitle(f"{ursi}, lat: {coords['lat']}, long: {coords['lat']}, {sunrise=}, {sunset=}")

    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()

    ax[0][0].set_ylim(None, 10)
    ax[0][1].set_ylim(None, 10)
    ax[1][0].set_ylim(None, 5)
    ax[1][1].set_ylim(None, 5)

    ax[0][0].set_title(
        f"r_sun={r1_sun}, gap_sun={gap_f0f2_sun}",
        fontsize=15,
    )
    ax[0][1].set_title(
        f"r_moon={r1_moon}, gap_moon={gap_f0f2_moon}",
        fontsize=15,
    )
    ax[1][0].set_title(
        f"r_sun={r2_sun}, gap_sun={gap_k_sun}",
        fontsize=15,
    )
    ax[1][1].set_title(
        f"r_moon={r2_moon}, gap_moon={gap_k_moon}",
        fontsize=15,
    )

    plot_graph(
        ax=ax[0][0],
        x_ax=hour_sun,
        y_ax=f0f2_sun,
        x_label='hour_sun',
        y_label='$f_0F_2$_sun',
        title='green: real',
        regression=False,
        solid=True,
    )
    plot_graph(
        ax=ax[0][0],
        x_ax=hour_sun,
        y_ax=jmodel_f0f2_sun,
        x_label='hour_sun',
        y_label='$f_0F_2$_sun',
        title='blue: model',
        color='blue',
        edgecolor='purple',
        regression=False,
        moon=True,
        solid=True,
    )

    plot_graph(
        ax=ax[0][1],
        x_ax=hour_moon,
        y_ax=f0f2_moon,
        x_label='hour_moon',
        y_label='$f_0F_2$_moon',
        title='green: real',
        regression=False,
        solid=True,
    )
    plot_graph(
        ax=ax[0][1],
        x_ax=hour_moon,
        y_ax=jmodel_f0f2_moon,
        x_label='hour_moon',
        y_label='$f_0F_2$_moon',
        title='blue: model',
        color='blue',
        edgecolor='purple',
        regression=False,
        moon=True,
        solid=True,
    )

    plot_graph(
        ax=ax[1][0],
        x_ax=hour_sun,
        y_ax=k_sun,
        x_label='hour_sun',
        y_label='k_sun',
        title='green: real',
        regression=False,
        solid=True,
    )
    plot_graph(
        ax=ax[1][0],
        x_ax=hour_sun,
        y_ax=jmodel_k_sun,
        x_label='hour_sun',
        y_label='k_sun',
        title='blue: model',
        color='blue',
        edgecolor='purple',
        regression=False,
        moon=True,
        solid=True,
    )

    plot_graph(
        ax=ax[1][1],
        x_ax=hour_moon,
        y_ax=k_moon,
        x_label='hour_moon',
        y_label='k_moon',
        title='green: real',
        regression=False,
        solid=True,
    )
    plot_graph(
        ax=ax[1][1],
        x_ax=hour_moon,
        y_ax=jmodel_k_moon,
        x_label='hour_moon',
        y_label='k_moon',
        title='blue: model',
        color='blue',
        edgecolor='purple',
        regression=False,
        moon=True,
        solid=True,
    )

if __name__ == '__main__':
    print(calc_f0f2_k_jmodel_gap_for_day('PA836', '2019-01-01'))