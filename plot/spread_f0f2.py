from plot.graph import plot_graph
from plot.jakowski import get_jmodel_k_spread_for_month
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from dal import select_coords_by_ursi
from dal.handlers import (
    get_f0f2_k_spread_for_month,
    get_f0f2_k_spread_for_summer_winter,
    get_f0f2_k_spread_for_year,
    get_adr_spread_for_month,
)


def plot_jmodel_k_spread_for_month(
    ursi: str,
    month: int,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    j_sun_k, j_moon_k = get_jmodel_k_spread_for_month(ursi, month, year)
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlim(x_lim[0], x_lim[1])
    ax[0].set_ylim(y_lim[0], y_lim[1])

    sns.histplot(j_sun_k, kde=True, ax=ax[0])
    sns.histplot(j_moon_k, kde=True, ax=ax[1])

    mu_win_sun, std_win_sun = norm.fit(j_sun_k)
    textstr_j_sun = '\n'.join((
    r'$k=%.2f$' % (mu_win_sun, ),
    r'$\sigma^2=%.2f$' % (std_win_sun, )))
    
    mu_win_moon, std_win_moon = norm.fit(j_moon_k)
    textstr_j_moon = '\n'.join((
    r'$k=%.2f$' % (mu_win_moon, ),
    r'$\sigma^2=%.2f$' % (std_win_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax[0].text(0.05, 0.95, textstr_j_sun, transform=ax[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1].text(0.05, 0.95, textstr_j_moon, transform=ax[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def plot_f0f2_k_spread_for_month(
    ursi: str,
    month: int,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    coords = select_coords_by_ursi(ursi)

    ion_sun_k, ion_moon_k, sat_sun_k, sat_moon_k = get_f0f2_k_spread_for_month(ursi, month, year)
    j_sun_k, j_moon_k = get_jmodel_k_spread_for_month(ursi, month, year)

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(15, 24))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year} Month: {month}",
        fontsize=18, y=0.98,
    )

    ax[0][0].set_title('Ion-Sun', fontsize=15)
    ax[0][1].set_title('Ion-Moon', fontsize=15)
    ax[1][0].set_title('Sat-Sun', fontsize=15)
    ax[1][1].set_title('Sat-Moon', fontsize=15)
    ax[2][0].set_title('Jakowski-Sun', fontsize=15)
    ax[2][1].set_title('Jakowski-Moon', fontsize=15)
    
    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    ax[2][0].grid()
    ax[2][1].grid()
    
    ax[0][0].set_xlim(x_lim[0], x_lim[1])
    ax[0][0].set_ylim(y_lim[0], y_lim[1])
    ax[0][1].set_xlim(x_lim[0], x_lim[1])
    ax[0][1].set_ylim(y_lim[0], y_lim[1])
    ax[1][0].set_xlim(x_lim[0], x_lim[1])
    ax[1][0].set_ylim(y_lim[0], y_lim[1])
    ax[1][1].set_xlim(x_lim[0], x_lim[1])
    ax[1][1].set_ylim(y_lim[0], y_lim[1])
    ax[2][0].set_xlim(x_lim[0], x_lim[1])
    ax[2][0].set_ylim(y_lim[0], y_lim[1])
    ax[2][1].set_xlim(x_lim[0], x_lim[1])
    ax[2][1].set_ylim(y_lim[0], y_lim[1])

    sns.histplot(ion_sun_k, kde=True, ax=ax[0][0])
    sns.histplot(ion_moon_k, kde=True, ax=ax[0][1])
    sns.histplot(sat_sun_k, kde=True, ax=ax[1][0])
    sns.histplot(sat_moon_k, kde=True, ax=ax[1][1])
    sns.histplot(j_sun_k, kde=True, ax=ax[2][0])
    sns.histplot(j_moon_k, kde=True, ax=ax[2][1])
    
    k_ion_sun, std_ion_sun = norm.fit(ion_sun_k)
    text_ion_sun = '\n'.join((
    r'$k=%.2f$' % (k_ion_sun, ),
    r'$\sigma^2=%.2f$' % (std_ion_sun, )))
    
    k_ion_moon, std_ion_moon = norm.fit(ion_moon_k)
    text_ion_moon = '\n'.join((
    r'$k=%.2f$' % (k_ion_moon, ),
    r'$\sigma^2=%.2f$' % (std_ion_moon, )))
    
    k_sat_sun, std_sat_sun = norm.fit(sat_sun_k)
    text_sat_sun = '\n'.join((
    r'$k=%.2f$' % (k_sat_sun, ),
    r'$\sigma^2=%.2f$' % (std_sat_sun, )))
    
    k_sat_moon, std_sat_moon = norm.fit(sat_moon_k)
    text_sat_moon = '\n'.join((
    r'$k=%.2f$' % (k_sat_moon, ),
    r'$\sigma^2=%.2f$' % (std_sat_moon, )))

    k_j_sun, std_j_sun = norm.fit(j_sun_k)
    text_j_sun = '\n'.join((
    r'$k=%.2f$' % (k_j_sun, ),
    r'$\sigma^2=%.2f$' % (std_j_sun, )))
    
    k_j_moon, std_j_moon = norm.fit(j_moon_k)
    text_j_moon = '\n'.join((
    r'$k=%.2f$' % (k_j_moon, ),
    r'$\sigma^2=%.2f$' % (std_j_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, text_ion_sun, transform=ax[0][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, text_ion_moon, transform=ax[0][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][0].text(0.05, 0.95, text_sat_sun, transform=ax[1][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, text_sat_moon, transform=ax[1][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[2][0].text(0.05, 0.95, text_j_sun, transform=ax[2][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[2][1].text(0.05, 0.95, text_j_moon, transform=ax[2][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def plot_f0f2_k_spread_for_summer_winter(
    ursi: str,
    year: int=2019,
    sat_tec: bool=False
):
    coords = select_coords_by_ursi(ursi)

    sum_result, win_result = count_f0f2_k_spreading_for_summer_winter(ursi, year)
    sum_sun_result, sum_moon_result = sum_result
    win_sun_result, win_moon_result = win_result

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year}",
        fontsize=18, y=0.98,
    )
    
    ax[0][0].set_title('Summer - Sun', fontsize=15)
    ax[0][1].set_title('Summer - Moon', fontsize=15)
    ax[1][0].set_title('Winter - Sun', fontsize=15)
    ax[1][1].set_title('Winter - Moon', fontsize=15)
    
    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    
    ax[0][0].set_xlim(None, 10)
    ax[0][0].set_ylim(None, 30)
    ax[0][1].set_xlim(None, 10)
    ax[0][1].set_ylim(None, 30)
    ax[1][0].set_xlim(None, 10)
    ax[1][0].set_ylim(None, 30)
    ax[1][1].set_xlim(None, 10)
    ax[1][1].set_ylim(None, 30)

    sns.histplot(sum_sun_result, kde=True, ax=ax[0][0])
    sns.histplot(sum_moon_result, kde=True, ax=ax[0][1])
    sns.histplot(win_sun_result, kde=True, ax=ax[1][0])
    sns.histplot(win_moon_result, kde=True, ax=ax[1][1])
    
    
    mu_sum_sun, std_sum_sun = norm.fit(sum_sun_result)
    textstr_sum_sun = '\n'.join((
    r'$k=%.2f$' % (mu_sum_sun, ),
    r'$\sigma^2=%.2f$' % (std_sum_sun, )))
    
    mu_sum_moon, std_sum_moon = norm.fit(sum_moon_result)
    textstr_sum_moon = '\n'.join((
    r'$k=%.2f$' % (mu_sum_moon, ),
    r'$\sigma^2=%.2f$' % (std_sum_moon, )))
    
    mu_win_sun, std_win_sun = norm.fit(win_sun_result)
    textstr_win_sun = '\n'.join((
    r'$k=%.2f$' % (mu_win_sun, ),
    r'$\sigma^2=%.2f$' % (std_win_sun, )))
    
    mu_win_moon, std_win_moon = norm.fit(win_moon_result)
    textstr_win_moon = '\n'.join((
    r'$k=%.2f$' % (mu_win_moon, ),
    r'$\sigma^2=%.2f$' % (std_win_moon, )))


    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, textstr_sum_sun, transform=ax[0][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, textstr_sum_moon, transform=ax[0][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][0].text(0.05, 0.95, textstr_win_sun, transform=ax[1][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, textstr_win_moon, transform=ax[1][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def plot_f0f2_k_spreading_for_year(ursi: str, year: int=2019):
    coords = select_coords_by_ursi(ursi)
    
    sun_range, moon_range = count_f0f2_k_spreading_for_year(ursi, year)
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year}",
        fontsize=18, y=0.98,
    )
    
    
    ax[0].set_title('Sun', fontsize=15)
    ax[1].set_title('Moon', fontsize=15)
    ax[0].grid()
    ax[1].grid()
    
    ax[0].set_xlim(None, 10)
    ax[0].set_ylim(None, 50)
    ax[1].set_xlim(None, 10)
    ax[1].set_ylim(None, 50)

    sns.histplot(sun_range, kde=True, ax=ax[0])
    sns.histplot(moon_range, kde=True, ax=ax[1])
    
    mu_sun, std_sun = norm.fit(sun_range)
    textstr_sun = '\n'.join((
    r'$k=%.2f$' % (mu_sun, ),
    r'$\sigma^2=%.2f$' % (std_sun, )))
    
    mu_moon, std_moon = norm.fit(moon_range)
    textstr_moon = '\n'.join((
    r'$k=%.2f$' % (mu_moon, ),
    r'$\sigma^2=%.2f$' % (std_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0].text(0.05, 0.95, textstr_sun, transform=ax[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1].text(0.05, 0.95, textstr_moon, transform=ax[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)










def plot_k_spreading_lat_split_month_graph(month: int, year: int, stations_list: list[str]):
    k_sun_range = []
    k_moon_range = []
    lat_range = []

    for s in stations_list:
        try:
            k = calc_f0f2_k_mean_for_month(s, month, year)

            k_sun_range.append(sum(k[0])/len(k[0]))
            k_moon_range.append(sum(k[1])/len(k[1]))
            lat_range.append(select_coords_by_ursi(s)['lat'])
        except Exception as ex:
            print(ex)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,6))
    
    fig.suptitle(f"Year: {year}, Month: {month}", fontsize=20, y=0.96)

    ax[0].grid()
    ax[1].grid()
    
    ax[0].set_xlim(None, 60)
    ax[0].set_ylim(None, 10)
    ax[1].set_xlim(None, 60)
    ax[1].set_ylim(None, 10)

    plot_graph(
        ax[0], lat_range, k_sun_range,
        'lat', 'k', 'Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[1], lat_range, k_moon_range,
        'lat', 'k', 'Moon', color='purple',
        edgecolor='b', const=True,
    )


def plot_k_spreading_lat_sum_win_split_graph(month: int, year: int, stations_list: list[str]):
    k_sum_sun_range = []
    k_sum_moon_range = []
    k_win_sun_range = []
    k_win_moon_range = []
    lat_range = []

    for s in stations_list:
        try:
            k = calc_f0f2_k_mean_for_summer_winter(s, year)

            k_sum_sun_range.append(k[0][0])
            k_sum_moon_range.append(k[0][1])
            k_win_sun_range.append(k[1][0])
            k_win_moon_range.append(k[1][1])

            lat_range.append(select_coords_by_ursi(s)['lat'])
        except Exception as ex:
            print(ex)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15,6))
    
    fig.suptitle(f"Year: {year}, Month: {month}", fontsize=20, y=0.96)

    ax[0][0].grid()
    ax[0][1].grid()
    ax[1][0].grid()
    ax[1][1].grid()

    ax[0][0].set_xlim(None, 60)
    ax[0][1].set_ylim(None, 10)
    ax[1][0].set_xlim(None, 60)
    ax[1][1].set_ylim(None, 10)

    plot_graph(
        ax[0][0], lat_range, k_sum_sun_range,
        'lat', 'k', 'Sum-Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[0][1], lat_range, k_sum_moon_range,
        'lat', 'k', 'Win-Moon', color='purple',
        edgecolor='b', const=True,
    )
    plot_graph(
        ax[1][0], lat_range, k_win_sun_range,
        'lat', 'k', 'Sum-Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[1][1], lat_range, k_win_moon_range,
        'lat', 'k', 'Win-Moon', color='purple',
        edgecolor='b', const=True,
    )
