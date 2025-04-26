from dal.models import (
    select_2h_avr_for_day_with_sat_tec,
    select_ion_tec_sat_tec,
    select_coords_by_ursi,
)
from dal.handlers import (
    get_adr_spread_for_month,
)
from plot.jakowski import split_to_sun_moon
from plot.utils import (
    cast_data_to_dataframe,
    get_month_days_count,
    get_sunrise_sunset,
    make_linear_regression,
    split_df_to_sun_moon,
)
from scipy.stats import pearsonr
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt


def calc_ion_sat_adr_mean_for_day(
    ursi: str,
    date: str,
):
    row_data = select_ion_tec_sat_tec(ursi, date)
    ion_tec = [r[1] for r in row_data]
    sat_tec = [r[2] for r in row_data]

    mlr = lambda x, y: make_linear_regression(y=y, x=x, const=True)
    reg = mlr(ion_tec, sat_tec)
    a, a_err = reg.params[1], reg.bse[1]
    d, d_err = reg.params[0], reg.bse[0]

    r = round(pearsonr(sat_tec, ion_tec)[0], 2)

    return {
        'a': a,
        'a_err': a_err,
        'd': d,
        'd_err': d_err,
        'r': r,
    }


# def get_jmodel_k_spread_for_month(
#     ursi: str,
#     month: int,
#     year: int,
# ):
#     jk_sun = []
#     jk_moon = []

#     for d in range(1, get_month_days_count(month) + 1):
#         day = f'0{d}' if d < 10 else d
#         date = f'{year}-{month}-{day}'

#         jk = select_ion_tec_sat_tec(ursi, date)
#         sun , moon = split_to_sun_moon(jk, ursi, date)
#         jk_sun = [*jk_sun, *sun]
#         jk_moon = [*jk_moon, *moon]
    


def calc_f0f2_k_mean_for_day(
    ursi: str,
    date: str,
):
    data = select_2h_avr_for_day_with_sat_tec(ursi, date)
    df = cast_data_to_dataframe(
        data,
        columns=['hour', 'f0f2', 'ion_tec', 'sat_tec', 'b0'],
        sat_tec=True,
    )

    sun, moon = split_df_to_sun_moon(df, ursi, date)

    mlr = lambda df, tec_name: make_linear_regression(
        y=[v**2 for v in df['f0f2']],
        x=df[tec_name],
        )
    
    ion_reg_sun = mlr(sun, 'ion_tec')
    ion_reg_moon = mlr(moon, 'ion_tec')
    ion_sun_k, ion_sun_k_err = ion_reg_sun.params[0], ion_reg_sun.bse[0]
    ion_moon_k, ion_moon_k_err = ion_reg_moon.params[0], ion_reg_moon.bse[0]

    sat_reg_sun = mlr(sun, 'sat_tec')
    sat_reg_moon = mlr(moon, 'sat_tec')
    sat_sun_k, sat_sun_k_err = sat_reg_sun.params[0], sat_reg_sun.bse[0]
    sat_moon_k, sat_moon_k_err = sat_reg_moon.params[0], sat_reg_moon.bse[0]

    return {
        'ion': {
            'sun': {'k': ion_sun_k, 'err': ion_sun_k_err},
            'moon': {'k': ion_moon_k, 'err': ion_moon_k_err},
        },
        'sat': {
            'sun': {'k': sat_sun_k, 'err': sat_sun_k_err},
            'moon': {'k': sat_moon_k, 'err': sat_moon_k_err},
        },
    }


def calc_b0_ab_mean_for_day(
    ursi: str,
    date: str,
):
    data = select_2h_avr_for_day_with_sat_tec(ursi, date)
    df = cast_data_to_dataframe(
        data,
        columns=['hour', 'f0f2', 'ion_tec', 'sat_tec', 'b0'],
        sat_tec=True,
    )

    sun, moon = split_df_to_sun_moon(df, ursi, date)

    mlr = lambda df, tec_name, turn: make_linear_regression(
        y=[v**2 for v in df['b0']],
        x=df[tec_name],
        const=True,
        turn=turn,
    )

    sat_reg_sun = mlr(sun, 'sat_tec', False)
    sat_reg_moon = mlr(moon, 'sat_tec', True)
    sat_sun_a, sat_sun_a_err = sat_reg_sun.params[1], sat_reg_sun.bse[1]
    sat_sun_b, sat_sun_b_err = sat_reg_sun.params[0], sat_reg_sun.bse[0]
    sat_moon_a, sat_moon_a_err = sat_reg_moon.params[1], sat_reg_moon.bse[1]
    sat_moon_b, sat_moon_b_err = sat_reg_moon.params[0], sat_reg_moon.bse[0]

    return {
            'sun': {
                'a': sat_sun_a, 'a_err': sat_sun_a_err,
                'b': sat_sun_b, 'b_err': sat_sun_b_err,
            },
            'moon': {
                'a': sat_moon_a, 'a_err': sat_moon_a_err,
                'b': sat_moon_b, 'b_err': sat_moon_b_err,
            },
        }


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

    ax[0].set_title('Tang', fontsize=15)
    ax[1].set_title('Diff betw ion and sat $TEC$', fontsize=15)
    ax[2].set_title('Correlation coefficient', fontsize=15)
    
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

    ax[0].set_title('Tang', fontsize=15)
    ax[1].set_title('Diff betw ion and sat $TEC$', fontsize=15)
    ax[2].set_title('Correlation coefficient', fontsize=15)
    
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