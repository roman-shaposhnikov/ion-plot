from plot.graph import plot_graph
from plot.utils import (
    get_month_days_count,
)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from dal import select_coords_by_ursi
from dal.handlers import (
    get_b0_ab_spread_for_month,
    get_b0_ab_spread_for_summer_winter,
    get_b0_ab_spread_for_year,
)


def plot_b0f2_ab_spread_for_month(
    ursi: str,
    month: int,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    coords = select_coords_by_ursi(ursi)
    sat_sun_k, sat_moon_k = get_b0_ab_spread_for_month(ursi, month, year)

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(15, 6))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year} Month: {month}",
        fontsize=18, y=0.98,
    )

    ax[0][0].set_title('Ion-Sun', fontsize=15)
    ax[0][1].set_title('Ion-Moon', fontsize=15)
    ax[1][0].set_title('Sat-Sun', fontsize=15)
    ax[1][1].set_title('Sat-Moon', fontsize=15)
    
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

    sns.histplot(ion_sun_k, kde=True, ax=ax[0][0])
    sns.histplot(ion_moon_k, kde=True, ax=ax[0][1])
    sns.histplot(sat_sun_k, kde=True, ax=ax[1][0])
    sns.histplot(sat_moon_k, kde=True, ax=ax[1][1])

    mu_sum_sun, std_sum_sun = norm.fit(ion_sun_k)
    textstr_ion_sun = '\n'.join((
    r'$k=%.2f$' % (mu_sum_sun, ),
    r'$\sigma^2=%.2f$' % (std_sum_sun, )))
    
    mu_sum_moon, std_sum_moon = norm.fit(ion_moon_k)
    textstr_ion_moon = '\n'.join((
    r'$k=%.2f$' % (mu_sum_moon, ),
    r'$\sigma^2=%.2f$' % (std_sum_moon, )))
    
    mu_win_sun, std_win_sun = norm.fit(sat_sun_k)
    textstr_sat_sun = '\n'.join((
    r'$k=%.2f$' % (mu_win_sun, ),
    r'$\sigma^2=%.2f$' % (std_win_sun, )))
    
    mu_win_moon, std_win_moon = norm.fit(sat_moon_k)
    textstr_sat_moon = '\n'.join((
    r'$k=%.2f$' % (mu_win_moon, ),
    r'$\sigma^2=%.2f$' % (std_win_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, textstr_ion_sun, transform=ax[0][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, textstr_ion_moon, transform=ax[0][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][0].text(0.05, 0.95, textstr_sat_sun, transform=ax[1][0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, textstr_sat_moon, transform=ax[1][1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def count_b0_t_spreading_for_month(
    ursi: str,
    month: int,
    year: int=2019,
) -> list[float]:
    sun_result: list[float] = []
    moon_result: list[float] = []
    
    for day in range(1, get_month_days_count(month) + 1):
        str_month = f'0{month}' if month < 10 else f'{month}'
        str_day = f'0{day}' if day < 10 else f'{day}'
        date = f"{year}-{str_month}-{str_day}"
        
        try:
            t = count_b0_t_for_day(ursi, date)

            sun_result.append(t[0])
            moon_result.append(t[1])
        except Exception as ex:
            print(ex)

    return sun_result, moon_result


def count_b0_t_spreading_for_summer_winter(
    ursi: str,
    year: int=2019,
) -> list[float]:
    sum_sun_result: list[float] = []
    sum_moon_result: list[float] = []
        
    win_sun_result: list[float] = []
    win_moon_result: list[float] = []

    if select_coords_by_ursi(ursi)['lat'] > 0:
        sum_month = north_summer
        win_month = north_winter
    else:
        sum_month = north_winter
        win_month = north_summer
    
    for m in sum_month:
        try:
            sun, moon = count_b0_t_spreading_for_month(ursi, m, year)
            sum_sun_result = [*sun, *sum_sun_result]
            sum_moon_result = [*moon, *sum_moon_result]
        except:
            pass

    for m in win_month:
        try:
            sun, moon = count_b0_t_spreading_for_month(ursi, m, year)
            win_sun_result = [*sun, *win_sun_result]
            win_moon_result = [*moon, *win_moon_result]
        except:
            pass
        
    return (sum_sun_result, sum_moon_result), (win_sun_result, win_moon_result)


def count_b0_t_spreading_for_year(ursi: str, year: int=2019) -> list[float]:
    sun_result: list[float] = []
    moon_result: list[float] = []
        
    for m in range(1, 13):
        try:
            sun, moon = count_b0_t_spreading_for_month(ursi, m, year)
            sun_result = [*sun_result, *sun]
            moon_result = [*moon_result, *moon]
        except:
            pass
    
    return sun_result, moon_result


def calc_b0_ab_mean_for_month(
    ursi: str,
    month: int,
    year: int,
):
    sun_range, moon_range = count_b0_t_spreading_for_month(ursi, month, year)
    
    sun_mean, sun_std_err = norm.fit(sun_range)
    moon_mean, moon_std_err = norm.fit(moon_range)

    return ((sun_mean, sun_std_err), (moon_mean, moon_std_err))


def calc_b0_t_mean_for_summer_winter(
    ursi: str,
    year: int=2019 ,
) -> tuple[tuple[float]]:
    sum_range, win_range = count_b0_t_spreading_for_summer_winter(ursi, year)
    
    sum_sun_mean, sum_sun_std_err = norm.fit(sum_range[0])
    sum_moon_mean, sum_moon_std_err = norm.fit(sum_range[1])
    win_sun_mean, win_sun_std_err = norm.fit(win_range[0])
    win_moon_mean, win_moon_std_err = norm.fit(win_range[1])
    
    return (
        ((sum_sun_mean, sum_sun_std_err), (sum_moon_mean, sum_moon_std_err)),
        ((win_sun_mean, win_sun_std_err), (win_moon_mean, win_moon_std_err)),
    )


def calc_b0_t_mean_for_year(
    ursi: str,
    year: int=2019 ,
) -> tuple[float]:
    sun_range, moon_range = count_b0_t_spreading_for_year(ursi, year)
    
    sun_mean, sun_std_err = norm.fit(sun_range)
    moon_mean, moon_std_err = norm.fit(moon_range)

    return ((sun_mean, sun_std_err), (moon_mean, moon_std_err))


def plot_b0_t_spreading_for_month(
    ursi: str,
    month: int,
    year: int=2019,
):
    coords = select_coords_by_ursi(ursi)
    k_sun_range, k_moon_range = count_b0_t_spreading_for_month(ursi, month, year)
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year} Month: {month}",
        fontsize=18, y=0.98,
    )

    ax[0].set_title('Sun', fontsize=15)
    ax[1].set_title('Moon', fontsize=15)
    ax[0].grid()
    
    ax[1].grid()
    
    ax[0].set_xlim(None, 10)
    ax[0].set_ylim(None, 12)
    ax[1].set_xlim(None, 10)
    ax[1].set_ylim(None, 12)

    sns.histplot(k_sun_range, kde=True, ax=ax[0])
    sns.histplot(k_moon_range, kde=True, ax=ax[1])
    
    mu_sun, std_sun = norm.fit(k_sun_range)
    textstr_sun = '\n'.join((
    r'$\mu=%.2f$' % (mu_sun, ),
    r'$\sigma^2=%.2f$' % (std_sun, )))
    
    mu_moon, std_moon = norm.fit(k_moon_range)
    textstr_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_moon, ),
    r'$\sigma^2=%.2f$' % (std_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0].text(0.05, 0.95, textstr_sun, transform=ax[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1].text(0.05, 0.95, textstr_moon, transform=ax[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def plot_b0_t_spreading_for_summer_winter(
    ursi: str,
    year: int=2019,
):
    coords = select_coords_by_ursi(ursi)
    
    sum_result, win_result = count_b0_t_spreading_for_summer_winter(ursi, year)
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
    r'$\mu=%.2f$' % (mu_sum_sun, ),
    r'$\sigma^2=%.2f$' % (std_sum_sun, )))
    
    mu_sum_moon, std_sum_moon = norm.fit(sum_moon_result)
    textstr_sum_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_sum_moon, ),
    r'$\sigma^2=%.2f$' % (std_sum_moon, )))
    
    mu_win_sun, std_win_sun = norm.fit(win_sun_result)
    textstr_win_sun = '\n'.join((
    r'$\mu=%.2f$' % (mu_win_sun, ),
    r'$\sigma^2=%.2f$' % (std_win_sun, )))
    
    mu_win_moon, std_win_moon = norm.fit(win_moon_result)
    textstr_win_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_win_moon, ),
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


def plot_b0_t_spreading_for_year(ursi: str, year: int=2019):
    coords = select_coords_by_ursi(ursi)
    
    sun_range, moon_range = count_b0_t_spreading_for_year(ursi, year)
    
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
    r'$\mu=%.2f$' % (mu_sun, ),
    r'$\sigma^2=%.2f$' % (std_sun, )))
    
    mu_moon, std_moon = norm.fit(moon_range)
    textstr_moon = '\n'.join((
    r'$\mu=%.2f$' % (mu_moon, ),
    r'$\sigma^2=%.2f$' % (std_moon, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0].text(0.05, 0.95, textstr_sun, transform=ax[0].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax[1].text(0.05, 0.95, textstr_moon, transform=ax[1].transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


def plot_t_spreading_lat_split_month_graph(month: int, year: int, stations_list: list[str]):
    t_sun_range = []
    t_moon_range = []
    lat_range = []

    for s in stations_list:
        try:
            t = calc_b0_t_mean_for_month(s, month, year)

            t_sun_range.append(t[0])
            t_moon_range.append(t[1])
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
        ax[0], lat_range, t_sun_range,
        'lat', 't', 'Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[1], lat_range, t_moon_range,
        'lat', 't', 'Moon', color='purple',
        edgecolor='b', const=True,
    )


def plot_t_spreading_lat_sum_win_split_graph(month: int, year: int, stations_list: list[str]):
    t_sum_sun_range = []
    t_sum_moon_range = []
    t_win_sun_range = []
    t_win_moon_range = []
    lat_range = []

    for s in stations_list:
        try:
            t = calc_b0_t_mean_for_summer_winter(s, year)

            t_sum_sun_range.append(t[0][0])
            t_sum_moon_range.append(t[0][1])
            t_win_sun_range.append(t[1][0])
            t_win_moon_range.append(t[1][1])

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
        ax[0][0], lat_range, t_sum_sun_range,
        'lat', 't', 'Sum-Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[0][1], lat_range, t_sum_moon_range,
        'lat', 't', 'Win-Moon', color='purple',
        edgecolor='b', const=True,
    )
    plot_graph(
        ax[1][0], lat_range, t_win_sun_range,
        'lat', 't', 'Sum-Sun', color='orange',
        edgecolor='r',const=True,
    )
    plot_graph(
        ax[1][1], lat_range, t_win_moon_range,
        'lat', 't', 'Win-Moon', color='purple',
        edgecolor='b', const=True,
    )
