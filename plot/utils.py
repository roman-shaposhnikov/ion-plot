from datetime import datetime
from calendar import monthrange

import pandas as pd
import statsmodels.api as sm

from suntime import Sun, SunTimeException
from statsmodels.regression.linear_model import (
    RegressionResultsWrapper as RRW
)

from dal import select_coords_by_ursi
from dal.models import(
    IonData,
    transform_ion_data,
    transform_sat_data,
    select_middle_lat_stations,
)

north_summer = [5, 6, 7, 8, 9, 10]
north_winter = [11, 12, 1, 2, 3, 4]

#from astral import LocationInfo
#from astral.sun import sun
#from astral.location import Location
#
#
#l = LocationInfo('', '', '', 30.4, 262.3)
## l.timezone = 'US/Central'
## l.latitude = 30.4
## l.longitude = 262.3
#
#print(l)
#
## s = sun(city.observer, date=datetime.date(2009, 4, 22))
## print((
##     f'Dawn:    {s["dawn"]}\n'
#
##     f'Sunrise: {s["sunrise"]}\n'
##     f'Noon:    {s["noon"]}\n'
##     f'Sunset:  {s["sunset"]}\n'
##     f'Dusk:    {s["dusk"]}\n'
## ))


def get_sunrise_sunset(date, coords):
    sun = Sun(coords['lat'], coords['long'])
    abd = datetime.strptime(date, '%Y-%m-%d').date()
    
    try:
        sunrise = sun.get_sunrise_time(abd).time().strftime('%H')
        sunset = sun.get_sunset_time(abd).time().strftime('%H')

        return sunrise, sunset
    except SunTimeException as e:
        print(f"Error: {e}")





def cast_data_to_dataframe(
    data: IonData,
    columns: list[str],
    sat_tec: bool=False
) -> pd.DataFrame:
    return pd.DataFrame(
        transform_sat_data(data) if sat_tec else transform_ion_data(data),
        columns=columns
    )


def split_df_to_sun_moon(df, ursi, date):
    sunrise, sunset = get_sunrise_sunset(date, select_coords_by_ursi(ursi))
    hour = df['hour']
    
    if sunrise < sunset:
        sun = df[(hour >= sunrise) & (hour < sunset)]
        moon = df[(hour < sunrise) | (hour >= sunset)]
    else:
        sun = df[(hour >= sunrise) | (hour < sunset)]
        moon = df[(hour < sunrise) & (hour >= sunset)]
        
    return sun, moon


def convert_iso_to_day_of_year(date: str) -> int:
    return datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday


def get_month_days_count(month: int, year: int=2019) -> int:
    return monthrange(year, month)[1]


def make_linear_regression(
        y: list[float],
        x: list[float],
        const: bool=False,
        turn: bool=False,
) -> RRW:
    if turn:
        x, y = y, x
    if const: x = sm.add_constant(x)

    reg = sm.OLS(y, x).fit()

    return reg

