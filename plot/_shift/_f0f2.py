import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.axis import Axis as Ax
from pandas import DataFrame as DF

from dal.models import (
    select_coords_by_ursi,
    select_hour_avr_for_day,
    select_2h_avr_for_day_with_sat_tec,
)

from plot.graph import plot_squared_graph
from plot.utils import (
        cast_data_to_dataframe,
        get_month_days_count,
        get_sunrise_sunset,
)


def plot_tec_f0f2_graph(
        sun: DF,
        moon: DF,
        date: str,
        split: bool=True,
        xlim=(None, 15),
        ylim=(None, 30),
        regression: bool=True,
        const: bool=False,
        sat_tec: bool=False,
):
    x_name = 'sat_tec' if sat_tec else 'ion_tec'

    if not split:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        plot_squared_graph(
            ax, pd.concat([sun, moon]),
            x_name, 'f0f2', 'TEC', '$f_0F_2$',
            date, regression=regression, const=const,
        )
        return ax

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

    ax[0].set_xlim(xlim[0], xlim[1])
    ax[0].set_ylim(ylim[0], ylim[1])
    ax[1].set_xlim(xlim[0], xlim[1])
    ax[1].set_ylim(ylim[0], ylim[1])

    plot_squared_graph(
        ax[0], sun, x_name, 'f0f2', 'TEC', '$f_0F_2$',
        'Sun ' + date, color='orange', edgecolor='r',
        regression=regression, const=const,
    )
    plot_squared_graph(
        ax[1], moon, x_name, 'f0f2', 'TEC', '$f_0F_2$',
        'Moon ' + date, color='purple', edgecolor='b',
        regression=regression, const=const,
    )
    return ax


def subplot_tec_f0f2_graph(
        sun: DF,
        moon: DF,
        date: str,
        ax: Ax,
        split: bool=True,
        xlim=(None, 15),
        ylim=(None, 30),
        regression: bool=True,
        const: bool=False,
        sat_tec: bool=False,
) -> Ax:
    x_name = 'sat_tec' if sat_tec else 'ion_tec'

    if not split:
        ax = plot_squared_graph(
            ax, pd.concat([sun, moon]),
            x_name, 'f0f2', 'TEC', '$f_0F_2$', date,
            regression=regression, const=const,
        )
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        return ax

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    ax = plot_squared_graph(
        ax, sun, x_name, 'f0f2', 'TEC', '$f_0F_2$',
        'Sun ' + date, color='orange', edgecolor='r',
        regression=regression, const=const,
    )
    ax = plot_squared_graph(
        ax, moon, x_name, 'f0f2', 'TEC', '$f_0F_2$',
        'Moon ' + date, color='purple', edgecolor='b',
        regression=regression, const=const,
    )
    ax.grid()

    return ax


def plot_tec_f0f2_for_day_graph(
    ursi: str,
    date: str,
    ax = None,
    split = True,
    xlim=(None, 15),
    ylim=(None, 30),
    regression: bool=True,
    const: bool=False,
    sat_tec: bool=False,
) -> None:
    if not sat_tec:
        df = cast_data_to_dataframe(
            select_hour_avr_for_day(ursi, date),
            columns=['hour', 'f0f2', 'ion_tec', 'b0'],
        )
    else:
        df = cast_data_to_dataframe(
            select_2h_avr_for_day_with_sat_tec(ursi, date),
            columns=['hour','f0f2', 'ion_tec', 'sat_tec', 'b0'],
            sat_tec=True,
        )

    sunrise, sunset = get_sunrise_sunset(date, select_coords_by_ursi(ursi))
    hour = df['hour']

    if sunrise < sunset:
        sun = df[(hour >= sunrise) & (hour < sunset)]
        moon = df[(hour < sunrise) | (hour >= sunset)]
    else:
        sun = df[(hour >= sunrise) | (hour < sunset)]
        moon = df[(hour < sunrise) & (hour >= sunset)]

    if ax != None:
        subplot_tec_f0f2_graph(
            sun, moon, date, ax, split, xlim,
            ylim, regression, const, sat_tec,
        )
    else:
        plot_tec_f0f2_graph(
            sun, moon, date, split, xlim, ylim,
            regression, const, sat_tec,
        )


def plot_tec_f0f2_for_each_day_in_month_graph(
    ursi: str,
    month: int,
    split=True,
    xlim=(None, 15),
    ylim=(None, 30),
    regression: bool=True,
    const: bool=False,
    sat_tec: bool=False,
) -> None:
    coords = select_coords_by_ursi(ursi)
    
    fig, ax_list = plt.subplots(ncols=3, nrows=11,figsize=(20, 60))
    fig.subplots_adjust(
        left=0.1,
        bottom=0.1,
        right=0.9,
        top=0.9,
        wspace=0.4,
        hspace=0.6,
    )
    
    axes = []
    for ax in ax_list:
        axes = [*axes, *ax]
    
    suptitle = f"{ursi}, lat: {coords['lat']} \
    long: {coords['long']}, Month: {month}"
    
    fig.suptitle(suptitle, fontsize=20, y=0.92)
    
    for day in range(1, get_month_days_count(month) + 1):
        str_month = f'0{month}' if month < 10 else f'{month}'
        str_day = f'0{day}' if day < 10 else f'{day}'
        try:
            plot_tec_f0f2_for_day_graph(
                ursi, f"2019-{str_month}-{str_day}",
                axes[day - 1], split, xlim, ylim,
                regression, const, sat_tec=sat_tec,
            )
        except Exception as ex:
            print(ex)
