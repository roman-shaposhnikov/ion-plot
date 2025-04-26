from dal.models import (
    select_coords_by_ursi,
    select_ad_mean_for_year,
    select_ion_tec_sat_tec,
)
from dal.handlers import (
    get_adr_spread_for_month,
    get_adr_spread_for_sum_win,
    get_adr_spread_for_year,
)
from plot.graph import plot_graph
from plot.utils import get_sunrise_sunset

from scipy.stats import pearsonr
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt


def plot_compare_ion_tec_sat_tec(ursi, date):
    coords = select_coords_by_ursi(ursi)
    sunrise, sunset = get_sunrise_sunset(date, coords)
    row_data = select_ion_tec_sat_tec(ursi, date)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

    ax[0].set_title('a', fontsize=15)
    ax[1].set_title('б', fontsize=15)

    ax[0].set_xlim(None, 15)
    ax[0].set_ylim(None, 15)
    ax[1].set_xlim(None, 15)
    ax[1].set_ylim(None, 15)

    hour = [r[0] for r in row_data]
    ion_tec = [r[1] for r in row_data]
    sat_tec = [r[2] for r in row_data]

    r = round(pearsonr(sat_tec, ion_tec)[0], 2)

    fig.suptitle(
        f"{ursi} lat: {coords['lat']}, long: {coords['lat']}, {date}\n\
sunrise={sunrise}, sunset={sunset}, r={r}",
        fontsize=15,
    )

    plot_graph(
        ax=ax[0],
        x_ax=ion_tec,
        y_ax=sat_tec,
        x_label='ion_tec',
        y_label='sat_tec',
        title=f"",
        regression=True,
        const=True,
        tang_name='a',
        const_name='d',
    )
    plot_graph(
        ax=ax[1],
        x_ax=hour,
        y_ax=ion_tec,
        x_label='hour',
        y_label='TEC',
        title=f"yellow: ion_tec",
        regression=False,
        solid=True,
    )
    plot_graph(
        ax=ax[1],
        x_ax=hour,
        y_ax=sat_tec,
        x_label='hour',
        y_label='TEC',
        title='blue: sat_tec',
        color='blue',
        edgecolor='purple',
        regression=False,
        moon=True,
        solid=True,
    )
    ax[1].grid()


def plot_adr_spread_for_month(
    ursi: str,
    month: int,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    coords = select_coords_by_ursi(ursi)
    a, d, r = get_adr_spread_for_month(ursi, month, year)

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 5))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year} Month: {month}",
        fontsize=18, y=0.99,
    )

    ax[0].set_title('Tang, a', fontsize=15)
    ax[1].set_title('Diff betw ion and sat $TEC$, d', fontsize=15)
    ax[2].set_title('Correlation coefficient, r', fontsize=15)
    
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    
    ax[0].set_xlim(x_lim[0], x_lim[1])
    ax[0].set_ylim(y_lim[0], y_lim[1])
    ax[1].set_xlim(x_lim[0], x_lim[1])
    ax[1].set_ylim(y_lim[0], y_lim[1])
    ax[2].set_xlim(x_lim[0], x_lim[1])
    ax[2].set_ylim(y_lim[0], y_lim[1])

    sns.histplot(a, kde=True, ax=ax[0])
    sns.histplot(d, kde=True, ax=ax[1])
    sns.histplot(r, kde=True, ax=ax[2])
    
    a, std_a = norm.fit(a)
    text_a = '\n'.join((
    r'$k=%.2f$' % (a, ),
    r'$\sigma^2=%.2f$' % (std_a, )))
    
    d, std_d = norm.fit(d)
    text_d = '\n'.join((
    r'$k=%.2f$' % (d, ),
    r'$\sigma^2=%.2f$' % (std_d, )))
    
    r, std_r = norm.fit(r)
    text_r = '\n'.join((
    r'$k=%.2f$' % (r, ),
    r'$\sigma^2=%.2f$' % (std_r, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0].text(0.05, 0.95, text_a, transform=ax[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1].text(0.05, 0.95, text_d, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[2].text(0.05, 0.95, text_r, transform=ax[2].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


def plot_adr_spread_for_sum_win(
    ursi: str,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    coords = select_coords_by_ursi(ursi)
    spread = get_adr_spread_for_sum_win(ursi, year)
    a_sum, d_sum, r_sum = spread[0]
    a_win, d_win, r_win = spread[1]

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year}",
        fontsize=18, y=0.99,
    )

    ax[0][0].set_title('Sum-a', fontsize=15)
    ax[0][1].set_title('Sum-d', fontsize=15)
    ax[0][2].set_title('Sum-r', fontsize=15)
    ax[1][0].set_title('Win-a', fontsize=15)
    ax[1][1].set_title('Win-d', fontsize=15)
    ax[1][2].set_title('Win-r', fontsize=15)
    
    ax[0][0].grid()
    ax[0][1].grid()
    ax[0][2].grid()
    ax[1][0].grid()
    ax[1][1].grid()
    ax[1][2].grid()

    ax[0][0].set_xlim(x_lim[0], x_lim[1])
    ax[0][0].set_ylim(y_lim[0], y_lim[1])
    ax[0][1].set_xlim(x_lim[0], x_lim[1])
    ax[0][1].set_ylim(y_lim[0], y_lim[1])
    ax[0][2].set_xlim(x_lim[0], x_lim[1])
    ax[0][2].set_ylim(y_lim[0], y_lim[1])

    ax[1][0].set_xlim(x_lim[0], x_lim[1])
    ax[1][0].set_ylim(y_lim[0], y_lim[1])
    ax[1][1].set_xlim(x_lim[0], x_lim[1])
    ax[1][1].set_ylim(y_lim[0], y_lim[1])
    ax[1][2].set_xlim(x_lim[0], x_lim[1])
    ax[1][2].set_ylim(y_lim[0], y_lim[1])

    sns.histplot(a_sum, kde=True, ax=ax[0][0])
    sns.histplot(d_sum, kde=True, ax=ax[0][1])
    sns.histplot(r_sum, kde=True, ax=ax[0][2])
    sns.histplot(a_win, kde=True, ax=ax[1][0])
    sns.histplot(d_win, kde=True, ax=ax[1][1])
    sns.histplot(r_win, kde=True, ax=ax[1][2])

    a_sum_mean, std_a_sum = norm.fit(a_sum)
    text_a_sum = '\n'.join((
    r'$a=%.2f$' % (a_sum_mean, ),
    r'$\sigma^2=%.2f$' % (std_a_sum, )))

    d_sum_mean, std_d_sum = norm.fit(d_sum)
    text_d_sum = '\n'.join((
    r'$d=%.2f$' % (d_sum_mean, ),
    r'$\sigma^2=%.2f$' % (std_d_sum, )))

    r_sum_mean, std_r_sum = norm.fit(r_sum)
    text_r_sum = '\n'.join((
    r'$r=%.2f$' % (r_sum_mean, ),
    r'$\sigma^2=%.2f$' % (std_r_sum, )))

    a_win_mean, std_a_win = norm.fit(a_win)
    text_a_win = '\n'.join((
    r'$a=%.2f$' % (a_win_mean, ),
    r'$\sigma^2=%.2f$' % (std_a_win, )))
    
    d_win_mean, std_d_win = norm.fit(d_win)
    text_d_win = '\n'.join((
    r'$d=%.2f$' % (d_win_mean, ),
    r'$\sigma^2=%.2f$' % (std_d_win, )))
    
    r_win_mean, std_r_win = norm.fit(r_win)
    text_r_win = '\n'.join((
    r'$r=%.2f$' % (r_win_mean, ),
    r'$\sigma^2=%.2f$' % (std_r_win, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0][0].text(0.05, 0.95, text_a_sum, transform=ax[0][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[0][1].text(0.05, 0.95, text_d_sum, transform=ax[0][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[0][2].text(0.05, 0.95, text_r_sum, transform=ax[0][2].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

    ax[1][0].text(0.05, 0.95, text_a_win, transform=ax[1][0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][1].text(0.05, 0.95, text_d_win, transform=ax[1][1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1][2].text(0.05, 0.95, text_r_win, transform=ax[1][2].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


def plot_adr_spread_for_year(
    ursi: str,
    year: int,
    x_lim=(None, 10),
    y_lim=(None, 20),
):
    coords = select_coords_by_ursi(ursi)
    a, d, r = get_adr_spread_for_year(ursi, year)

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 5))
    fig.suptitle(
        f"{ursi}, lat: {coords['lat']}  long: {coords['long']}, Year: {year}",
        fontsize=18, y=0.99,
    )

    ax[0].set_title('a', fontsize=15)
    ax[1].set_title('d', fontsize=15)
    ax[2].set_title('r', fontsize=15)
    
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    
    ax[0].set_xlim(x_lim[0], x_lim[1])
    ax[0].set_ylim(y_lim[0], y_lim[1])
    ax[1].set_xlim(x_lim[0], x_lim[1])
    ax[1].set_ylim(y_lim[0], y_lim[1])
    ax[2].set_xlim(x_lim[0], x_lim[1])
    ax[2].set_ylim(y_lim[0], y_lim[1])

    sns.histplot(a, kde=True, ax=ax[0])
    sns.histplot(d, kde=True, ax=ax[1])
    sns.histplot(r, kde=True, ax=ax[2])
    
    a, std_a = norm.fit(a)
    text_a = '\n'.join((
    r'$k=%.2f$' % (a, ),
    r'$\sigma^2=%.2f$' % (std_a, )))
    
    d, std_d = norm.fit(d)
    text_d = '\n'.join((
    r'$k=%.2f$' % (d, ),
    r'$\sigma^2=%.2f$' % (std_d, )))
    
    r, std_r = norm.fit(r)
    text_r = '\n'.join((
    r'$k=%.2f$' % (r, ),
    r'$\sigma^2=%.2f$' % (std_r, )))

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # place a text box in upper left in axes coords
    ax[0].text(0.05, 0.95, text_a, transform=ax[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[1].text(0.05, 0.95, text_d, transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax[2].text(0.05, 0.95, text_r, transform=ax[2].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


def plot_ad_lat_for_year(year: int, stations_list: list[str]):
    a_list = []
    d_list = []
    lat = []

    for s in stations_list:
        try:
            a, d, _ = select_ad_mean_for_year(s, year)[0]
            a_list.append(a)
            d_list.append(d)
            lat.append(select_coords_by_ursi(s)['lat'])
        except Exception as ex:
            print(ex)

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(10, 15))
    # fig.suptitle(f"Year: {year}", fontsize=18, y=0.99)

    ax[0].set_xlim(None, 65)
    ax[0].set_ylim(None, 10)
    ax[1].set_xlim(None, 65)
    ax[1].set_ylim(None, 10)
    print(f'{a_list=}')
    print(f'{d_list=}')

    plot_graph(
        ax=ax[0],
        x_ax=lat,
        y_ax=a_list,
        x_label='lat',
        y_label='a',
        title='',
        const=True,
    )
    plot_graph(
        ax=ax[1],
        x_ax=lat,
        y_ax=d_list,
        x_label='lat',
        y_label='d',
        title='',
        const=True,
        turn=True,
    )

