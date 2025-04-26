import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from typing import TypeAlias, Literal

from matplotlib.axis import Axis as Ax
from pandas import DataFrame as DF

from plot.utils import (
    cast_data_to_dataframe,
    get_month_days_count,
    get_sunrise_sunset,
    make_linear_regression,
)
from dal.models import (
    select_coords_by_ursi,
    select_hour_avr_for_day,
    select_2h_avr_for_day_with_sat_tec,
)


def set_title_date_k_err(ax: Ax, date: str, k: float, err: float) -> None:
    ax.set_title(
        f"{date}, k={round(k, 3)}, err={round(err, 3)}",
        fontsize=15,
    )


def plot_graph(
        ax: Ax,
        x_ax: list,
        y_ax: list,
        x_label: str,
        y_label: str,
        title: str,
        color: str='y',
        edgecolor: str='g',
        regression: bool=True,
        const: bool=False,
        turn: bool=False,
        moon: bool=False,
        solid: bool=False,
        tang_name: str='k',
        const_name: str='const',
) -> Ax:
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    if regression:
        if const:
            if turn:
                reg = make_linear_regression(x_ax, y_ax, const)
                linspace = np.linspace(0, max(y_ax), 100)
                ax.plot(reg.predict(sm.add_constant(linspace)), linspace, c=edgecolor)
            else:
                reg = make_linear_regression(y_ax, x_ax, const)
                linspace = np.linspace(0, max(x_ax), 100)
                ax.plot(linspace, reg.predict(sm.add_constant(linspace)), c=edgecolor)

            title =  title + f"{tang_name}={round(reg.params[1], 1)}, {tang_name}_err={round(reg.bse[1], 1)},\n\
{const_name}={round(reg.params[0], 1)}, {const_name}_err={round(reg.bse[0], 1)}"

            ax.text(
                0.05, 0.9 if moon else 0.95, title, transform=ax.transAxes,
                fontsize=15, verticalalignment='top', bbox=props,
            )

        else:
            reg = make_linear_regression(y_ax, x_ax, const)
            linspace = np.linspace(0, max(x_ax), 100)
            ax.plot(linspace, reg.predict(linspace), c=edgecolor)

            title = f"{title} {tang_name}={round(reg.params[0], 3)}, {tang_name}_err={round(reg.bse[0], 3)}"

            ax.text(
                0.05, 0.9 if moon else 0.95, title, transform=ax.transAxes,
                fontsize=15, verticalalignment='top', bbox=props,
            )
    else:
        title = f"{title}"
        ax.text(
            0.05, 0.9 if moon else 0.95, title, transform=ax.transAxes,
            fontsize=15, verticalalignment='top', bbox=props,
        )
    
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    if not solid:
        ax.scatter(x=x_ax, y=y_ax, marker='o', c=color, edgecolor=edgecolor)
    else:
        ax.plot(x_ax, y_ax, c=color)
    ax.grid()

    return ax


def plot_linear_graph(
        ax: Ax,
        df: DF,
        x_name: str,
        y_name: str,
        x_label: str,
        y_label: str,
        title: str,
        color: str='y',
        edgecolor: str='g',
        regression: bool=True,
        const: bool=False,
        turn: bool=False,
        moon: bool=False,
) -> Ax:
    x_ax = df[x_name]
    y_ax = df[y_name]

    return plot_graph(
            ax, x_ax, y_ax, x_label, y_label, title, color,
            edgecolor, regression, const, turn, moon,
    )


def plot_squared_graph(
        ax: Ax,
        df: DF,
        x_name: str,
        y_name: str,
        x_label: str,
        y_label: str,
        title: str,
        color: str='y',
        edgecolor: str='g',
        regression: bool=True,
        const: bool=False,
        turn: bool=False,
        moon: bool=False,
) -> Ax:
    x_ax = df[x_name]
    y_ax = [y**2 for y in df[y_name]]

    return plot_graph(
        ax, x_ax, y_ax, x_label, y_label, title, color,
        edgecolor, regression, const, turn, moon,
    )


f0f2Val = Literal['f0f2']
b0Val = Literal['b0']

def plot_tec_graph(
    value: f0f2Val | b0Val,
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
    if value == 'f0f2':
        y_label = '$f_0F_2^2$'
        graph = plot_squared_graph
    else:
        y_label = 'B0'
        graph = plot_linear_graph

    if not split:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        graph(
            ax=ax,
            df=pd.concat([sun, moon]),
            x_name=x_name,
            y_name=value,
            x_label='TEC',
            y_label=y_label,
            title=date,
            regression=regression,
            const=const,
        )
        return ax

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))

    ax[0].set_xlim(xlim[0], xlim[1])
    ax[0].set_ylim(ylim[0], ylim[1])
    ax[1].set_xlim(xlim[0], xlim[1])
    ax[1].set_ylim(ylim[0], ylim[1])

    graph(
        ax=ax[0],
        df=sun,
        x_name=x_name,
        y_name=value,
        x_label='TEC',
        y_label=y_label,
        title='Sun ' + date + '\n',
        color='orange',
        edgecolor='r',
        regression=regression,
        const=const,
    )
    graph(
        ax=ax[1],
        df=moon,
        x_name=x_name,
        y_name=value,
        x_label='TEC',
        y_label=y_label,
        title='Moon ' + date + '\n',
        color='purple',
        edgecolor='b',
        regression=regression,
        const=const,
        turn=(value == 'b0'),
        moon=True,
    )
    return ax


def subplot_tec_graph(
    value: f0f2Val | b0Val,
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
    if value == 'f0f2':
        y_label = '$f_0F_2^2$'
        graph = plot_squared_graph
    else:
        y_label = 'B0'
        graph = plot_linear_graph

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    if not split:
        ax = graph(
            ax=ax,
            df=pd.concat([sun, moon]),
            x_name=x_name,
            y_name=value,
            x_label='TEC',
            y_label=y_label,
            title=date,
            regression=regression,
            const=const,
        )
        return ax

    ax = graph(
        ax=ax,
        df=sun,
        x_name=x_name,
        y_name=value,
        x_label='TEC',
        y_label=y_label,
        title='Sun ' + date + '\n',
        color='orange',
        edgecolor='r',
        regression=regression,
        const=const,
    )
    ax = graph(
        ax=ax,
        df=moon,
        x_name=x_name,
        y_name=value,
        x_label='TEC',
        y_label=y_label,
        title='Moon ' + date + '\n',
        color='purple',
        edgecolor='b',
        regression=regression,
        const=const,
        turn=(value == 'b0'),
        moon=True,
    )
    ax.grid()

    return ax


def plot_tec_for_day_graph(
    value: f0f2Val | b0Val,
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

    sunrise, sunset = get_sunrise_sunset(
        date=date,
        coords=select_coords_by_ursi(ursi)
    )
    hour = df['hour']

    if sunrise < sunset:
        sun = df[(hour >= sunrise) & (hour < sunset)]
        moon = df[(hour < sunrise) | (hour >= sunset)]
    else:
        sun = df[(hour >= sunrise) | (hour < sunset)]
        moon = df[(hour < sunrise) & (hour >= sunset)]

    if ax != None:
        subplot_tec_graph(
            value=value,
            sun=sun,
            moon=moon,
            date=date,
            ax=ax,
            split=split,
            xlim=xlim,
            ylim=ylim,
            regression=regression,
            const=const,
            sat_tec=sat_tec,
        )
    else:
        plot_tec_graph(
            value=value,
            sun=sun,
            moon=moon,
            date=date,
            split=split,
            xlim=xlim,
            ylim=ylim,
            regression=regression,
            const=const,
            sat_tec=sat_tec,
        )


def plot_tec_for_each_day_in_month_graph(
    value: f0f2Val | b0Val,
    ursi: str,
    month: int,
    year: int,
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
            plot_tec_for_day_graph(
                value=value,
                ursi=ursi,
                date=f"{year}-{str_month}-{str_day}",
                ax=axes[day - 1],
                split=split,
                xlim=xlim,
                ylim=ylim,
                regression=regression,
                const=const,
                sat_tec=sat_tec,
            )
        except Exception as ex:
            print(ex)


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
    plot_tec_for_day_graph(
        'f0f2',
        ursi,
        date,
        ax,
        split,
        xlim,
        ylim,
        regression,
        const,
        sat_tec,
    )


def plot_tec_f0f2_for_each_day_in_month_graph(
    ursi: str,
    month: int,
    year: int,
    split=True,
    xlim=(None, 15),
    ylim=(None, 30),
    regression: bool=True,
    const: bool=False,
    sat_tec: bool=False,
) -> None:
    plot_tec_for_each_day_in_month_graph(
        'f0f2',
        ursi,
        month,
        year,
        split,
        xlim,
        ylim,
        regression,
        const,
        sat_tec,
    )


def plot_tec_b0_for_day_graph(
    ursi: str,
    date: str,
    ax = None,
    split = True,
    xlim=(None, 15),
    ylim=(None, 300),
    regression: bool=True,
    const: bool=False,
    sat_tec: bool=False,
) -> None:
    plot_tec_for_day_graph(
        'b0',
        ursi,
        date,
        ax,
        split,
        xlim,
        ylim,
        regression,
        const,
        sat_tec,
    )


def plot_tec_b0_for_each_day_in_month_graph(
    ursi: str,
    month: int,
    year: int,
    split=True,
    xlim=(None, 15),
    ylim=(None, 300),
    regression: bool=True,
    const: bool=False,
    sat_tec: bool=False,
) -> None:
    plot_tec_for_each_day_in_month_graph(
        'b0',
        ursi,
        month,
        year,
        split,
        xlim,
        ylim,
        regression,
        const,
        sat_tec,
    )