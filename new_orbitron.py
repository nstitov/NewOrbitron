import csv
import math
import os
import re
import sys
import threading
import time
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, NamedTuple

import win32api  # pip install pywin32==305, pip install pypiwin32
from ntplib import NTPClient
from pyorbital.orbital import Orbital
from PyQt5.QtWidgets import QApplication
from spacetrack import SpaceTrackClient
from tqdm import tqdm

from orbitron_exceptions import (NewOrbitronDataError, NewOrbitronIndexError,
                                 NewOrbitronSetupError)
from pyqt_radar import MainWindowRadar


class SatPosition(NamedTuple):
    xyz: list[float, float, float]
    dt: datetime


@dataclass(slots=True)
class CommParams:
    elevation: float
    azimuth: float
    downlink: float = None
    uplink: float = None


@dataclass(slots=True)
class SessionParams:
    start_session_dt: datetime
    end_session_dt: datetime

    start_elevation: float = None
    start_azimuth: float = None
    start_sun_azimuth: float = None
    start_sun_elevation: float = None

    max_elevation: float = None
    # Parameters values when elevation angle value is max
    max_azimuth: float = None
    max_session_dt: datetime = None
    max_sun_azimuth: float = None
    max_sun_elevation: float = None

    end_elevation: float = None
    end_azimuth: float = None
    end_sun_azimuth: float = None
    end_sun_elevation: float = None

    zero_crossing_azimuth_flag: bool = False


class Satellite:
    _MU = 398600.44e9
    _R_ECV = 6378.136e3
    _J_2 = 1082.627e-6
    _J_4 = -1.617608e-6
    _OMEGA_EARTH = 0.729211e-4

    def __init__(self, norad_id: int, uplink: float = None,
                 downlink: float = None, tle_data_folder: str = 'tle'):
        self.norad_id = norad_id
        self.uplink_freq = uplink
        self.downlink_freq = downlink

        self.tle_data_folder = tle_data_folder
        self.TLE_exist = False
        tle_data_dir = os.path.join(os.path.dirname(__file__), tle_data_folder)
        if not os.path.exists(tle_data_dir):
            os.makedirs(tle_data_dir)

        for file_name in os.listdir(tle_data_dir):
            if re.fullmatch(str(self.norad_id) + r'__\d{4}\-\d{2}\-\d{2}\.[3t]le', file_name):
                self.tle_file_name: str = file_name
                self.TLE_exist = True
                break
        else:
            print(f'Noone TLE file for satellite {self.norad_id} not found. The download will be performed.')
            self.update_tle()

        self.orbital = self._process_tle()

    def _process_tle(self) -> Orbital:
        '''Parse TLE file and transfer data to orbital paramaters for next preodiction center mass motion'''
        tle_data_dir = os.path.join(os.path.dirname(__file__), self.tle_data_folder)
        with open(os.path.join(tle_data_dir, self.tle_file_name), 'r', encoding='utf-8') as tle_file:
            tle_info = tle_file.read()
            tle_line0, tle_line1, tle_line2, *_ = tle_info.split('\n')
            self.satellite_name = tle_line0[2:]

        return Orbital('N', line1=tle_line1, line2=tle_line2)

    def _get_sat_position_eci(self, req_time: datetime) -> list[tuple[float, float, float]]:
        '''Get coordinate and speed of satellite center mass in ECI coordinate system at required time.'''
        pos, vel = self.orbital.get_position(req_time, normalize=False)

        return [(pos[0]*1000, pos[1]*1000, pos[2]*1000), (vel[0]*1000, vel[1]*1000, vel[2]*1000)]

    def _RP_centermass_ECI(self, x_0: float, y_0: float, z_0: float,
                           Vx_0: float, Vy_0: float, Vz_0: float) -> list[float]:
        '''Right part of differential equations with 4-th harmonic of Earth geopotential
        to propagate satellite center mass motion'''
        r: float = (x_0**2 + y_0**2 + z_0**2)**(0.5)

        mun = self._MU / r**2
        xn = x_0 / r
        yn = y_0 / r
        zn = z_0 / r
        an = self._R_ECV / r

        x = Vx_0
        y = Vy_0
        z = Vz_0
        Vx = -mun * xn - 1.5 * self._J_2 * mun * xn * an**2 * (1.0 - 5.0 * zn**2) + 0.625 * self._J_4 * mun * xn * an**4 * (3.0 + (63.0 * zn**2 - 42.0) * zn**2)
        Vy = -mun * yn - 1.5 * self._J_2 * mun * yn * an**2 * (1.0 - 5.0 * zn**2) + 0.625 * self._J_4 * mun * yn * an**4 * (3.0 + (63.0 * zn**2 - 42.0) * zn**2)
        Vz = -mun * zn - 1.5 * self._J_2 * mun * zn * an**2 * (3.0 - 5.0 * zn**2) + 0.625 * self._J_4 * mun * zn * an**4 * (15.0 + (63.0 * zn**2 - 70.0) * zn**2)

        return [x, y, z, Vx, Vy, Vz]

    def _propagate_centermass_ECI_RK4(self, pos_eci_init: tuple,
                                      vel_eci_init: tuple, step: float) -> list[tuple[float, float, float]]:
        '''Propagte satellite center mass motion by Runge Kutta method.'''
        step1_2: float = step / 2

        k_x_1, k_y_1, k_z_1, k_Vx_1, k_Vy_1, k_Vz_1 = self._RP_centermass_ECI(pos_eci_init[0],
                                                                              pos_eci_init[1],
                                                                              pos_eci_init[2],
                                                                              vel_eci_init[0],
                                                                              vel_eci_init[1],
                                                                              vel_eci_init[2])
        k_x_2, k_y_2, k_z_2, k_Vx_2, k_Vy_2, k_Vz_2 = self._RP_centermass_ECI(pos_eci_init[0] + step1_2 * k_x_1,
                                                                              pos_eci_init[1] + step1_2 * k_y_1,
                                                                              pos_eci_init[2] + step1_2 * k_z_1,
                                                                              vel_eci_init[0] + step1_2 * k_Vx_1,
                                                                              vel_eci_init[1] + step1_2 * k_Vy_1,
                                                                              vel_eci_init[2] + step1_2 * k_Vz_1)
        k_x_3, k_y_3, k_z_3, k_Vx_3, k_Vy_3, k_Vz_3 = self._RP_centermass_ECI(pos_eci_init[0] + step1_2 * k_x_2,
                                                                              pos_eci_init[1] + step1_2 * k_y_2,
                                                                              pos_eci_init[2] + step1_2 * k_z_2,
                                                                              vel_eci_init[0] + step1_2 * k_Vx_2,
                                                                              vel_eci_init[1] + step1_2 * k_Vy_2,
                                                                              vel_eci_init[2] + step1_2 * k_Vz_2)
        k_x_4, k_y_4, k_z_4, k_Vx_4, k_Vy_4, k_Vz_4 = self._RP_centermass_ECI(pos_eci_init[0] + step * k_x_3,
                                                                              pos_eci_init[1] + step * k_y_3,
                                                                              pos_eci_init[2] + step * k_z_3,
                                                                              vel_eci_init[0] + step * k_Vx_3,
                                                                              vel_eci_init[1] + step * k_Vy_3,
                                                                              vel_eci_init[2] + step * k_Vz_3)

        step_1_6 = step / 6
        x_0 = pos_eci_init[0] + step_1_6 * (k_x_1 + 2 * (k_x_2 + k_x_3) + k_x_4)
        y_0 = pos_eci_init[1] + step_1_6 * (k_y_1 + 2 * (k_y_2 + k_y_3) + k_y_4)
        z_0 = pos_eci_init[2] + step_1_6 * (k_z_1 + 2 * (k_z_2 + k_z_3) + k_z_4)
        Vx_0 = vel_eci_init[0] + step_1_6 * (k_Vx_1 + 2 * (k_Vx_2 + k_Vx_3) + k_Vx_4)
        Vy_0 = vel_eci_init[1] + step_1_6 * (k_Vy_1 + 2 * (k_Vy_2 + k_Vy_3) + k_Vy_4)
        Vz_0 = vel_eci_init[2] + step_1_6 * (k_Vz_1 + 2 * (k_Vz_2 + k_Vz_3) + k_Vz_4)

        return [(x_0, y_0, z_0), (Vx_0, Vy_0, Vz_0)]

    def _transform_eci_to_ecef(self, pos_eci: tuple[float, float, float],
                               GST: float, curr_date_seconds: float | int) -> list[float, float, float]:
        '''Transform coordanates from ECI coordinate system to ECEF coordinate system.'''
        S = GST + self._OMEGA_EARTH * curr_date_seconds

        x = pos_eci[0] * math.cos(S) + pos_eci[1] * math.sin(S)
        y = -pos_eci[0] * math.sin(S) + pos_eci[1] * math.cos(S)
        z = pos_eci[2]

        return [x, y, z]

    def _calculate_GMST(self, req_time: datetime) -> float:
        '''Calculate Greenwich Middle Sidereal Time.'''
        year = req_time.year - 1900
        month = req_time.month - 3
        if month < 0:
            month += 12
            year -= 1

        mjd = 15078 + 365 * year + int(year / 4) + int(0.5 + 30.6 * month)
        mjd += req_time.day + req_time.hour / 24 + req_time.minute / 1440 + req_time.second / 86400

        Tu = (math.floor(mjd) - 51544.5) / 36525.0
        GST = 1.753368559233266 + (628.3319706888409 + (6.770714e-6 - 4.51e-10 * Tu) * Tu) * Tu

        return GST

    def update_tle(self, token: dict[Literal['identity', 'password'], str] =
                   {'identity': 'godlike200040@gmail.com', 'password': 'SamaraUniversity2022'}) -> None:
        '''Download TLE files for satellites by SpaceTrack API.'''
        tle_data_dir: str = os.path.join(os.path.dirname(__file__), self.tle_data_folder)
        if self.TLE_exist:
            os.remove(os.path.join(tle_data_dir, self.tle_file_name))

        st: SpaceTrackClient = SpaceTrackClient(identity=token.get('identity'), password=token.get('password'))
        tle: str = st.tle_latest(norad_cat_id=self.norad_id, orderby='epoch desc', limit=1, format='3le')
        if not tle:
            print(f'No data for the {self.norad_id} satellite!')
        else:
            tle_info: str = tle.split('\n')[1]
            epoch_year: int = int(tle_info[18:20])
            epoch_day: int = int(tle_info[20:23])
            if epoch_year <= 50:
                epoch_year += 2000
            else:
                epoch_year += 1900

            epoch: datetime = datetime(epoch_year, 1, 1) + timedelta(days=epoch_day-1)
            self.tle_file_name: str = f'{self.norad_id}__{str(epoch.date())}.tle'
            with open(os.path.join(tle_data_dir, self.tle_file_name), 'w', encoding='utf-8') as tle_file:
                tle_file.write(tle)

            self.TLE_exist = True
            print(f'TLE file for {self.norad_id} satellite is downloaded (updated).')

    def predict_cm(self, start_dt: datetime = datetime.utcnow().replace(microsecond=0),
                   time_prediction: int = 86400, step_prediction: int = 1) -> None:
        '''Predict satellite center mass motion for required time prediction with required time step prediction in ECI
        coordinate system. After propagation transform coordinates from ECI coordinate system to ECEF coordinate system.
        By default used current utc datetime, propogation for one day and time step prediction 1 second.'''

        pos_ecef: list[SatPosition] = [None] * int(time_prediction / step_prediction)

        GST = self._calculate_GMST(start_dt)
        seconds_in_current_date = (start_dt - datetime(start_dt.year, start_dt.month, start_dt.day)).total_seconds()

        pos_eci, vel_eci = self._get_sat_position_eci(start_dt)
        pos_ecef_lst = self._transform_eci_to_ecef(pos_eci, GST, seconds_in_current_date)
        pos_ecef[0] = SatPosition(pos_ecef_lst, start_dt)
        current_dt = start_dt
        for i in tqdm(range(1, int(time_prediction / step_prediction)), desc=f'Center mass motion prediction for {self.norad_id} satellite', colour='red'):
            seconds_in_current_date += step_prediction
            current_dt += timedelta(seconds=step_prediction)
            pos_eci, vel_eci = self._propagate_centermass_ECI_RK4(pos_eci, vel_eci, step_prediction)
            pos_ecef_lst = self._transform_eci_to_ecef(pos_eci, GST, seconds_in_current_date)
            pos_ecef[i] = SatPosition(pos_ecef_lst, current_dt)

        self.pos_ecef = pos_ecef


class GroundStation:
    _R_ECV = 6378.136e3
    _R_POL = 6356.7523e3
    _E_SQUARE = 1 - _R_POL**2 / _R_ECV**2
    _F = 1 - _R_POL / _R_ECV

    def __init__(self, position: list[float, float, float] | tuple[float, float, float],
                 elevation_min: float, name: str = 'default'):
        '''position = [longitude (deg), latatide (deg), altitude (m)]'''
        self.geo_position = [math.radians(position[0]), math.radians(position[1]), position[2]]
        self.ecef_position = self._transform_geodetic_to_ecef()
        self.elevation_min = elevation_min
        self.name = name

    def _transform_geodetic_to_ecef(self) -> list[float, float, float]:
        '''Transform ground station coordinate from geodetic coordinate system to ECEF coordinate system.'''
        N = self._R_ECV / (1 - self._E_SQUARE * math.sin(self.geo_position[1])**2)**(0.5)
        x = (N + self.geo_position[2]) * math.cos(self.geo_position[1]) * math.cos(self.geo_position[0])
        y = (N + self.geo_position[2]) * math.cos(self.geo_position[1]) * math.sin(self.geo_position[0])
        z = ((1 - self._F)**2 * N + self.geo_position[2]) * math.sin(self.geo_position[1])

        return [x, y, z]


class SatelliteStationComm:
    _R_E   = 6371.302e3
    _R_ECV = 6378.136e3
    _ALF_CZJ = 1/298.257223563
    _c = 299792458
    _PREDICT_SESSION_DELAY = 300

    def __init__(self, satellite: Satellite, station: GroundStation):
        self.satellite: Satellite = satellite
        self.station: GroundStation = station
        self.session_params: list[SessionParams] = []
        self.comm_data_for_all_sat_prediction: dict[datetime, CommParams] = {}

    def _transform_ecef_to_geodetic(self, pos_ecef: tuple[float, float, float]) -> list[float, float, float]:
        '''Transform coordinate from ECEF coordinate system to geodetic coordinate system'''
        fi = math.atan2(pos_ecef[2], (pos_ecef[0]**2 + pos_ecef[1]**2)**(0.5))
        lam = math.atan2(pos_ecef[1], pos_ecef[0])
        r_g = (pos_ecef[0]**2 + pos_ecef[1]**2 + pos_ecef[2]**2)**(0.5)
        R_z = self._R_ECV * (1 - self._ALF_CZJ * math.sin(fi)**2)
        h = r_g - R_z

        return [lam, fi, h]

    def _calculate_comm_session_times_for_prediction_period(self) -> list[tuple[SatPosition, SatPosition]]:
        '''Define all communication session times with satellite in predicted
        satellite center mass motion period.'''
        session_times: list[tuple[SatPosition, SatPosition]] = []

        comm_flag = False
        for pos_sat in self.satellite.pos_ecef:
            r1 = [pos_sat.xyz[0] - self.station.ecef_position[0],
                  pos_sat.xyz[1] - self.station.ecef_position[1],
                  pos_sat.xyz[2] - self.station.ecef_position[2]]
            r2 = [self.station.ecef_position[0],
                  self.station.ecef_position[1],
                  self.station.ecef_position[2]]
            dot_r1r2 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2]
            mod_r1 = (r1[0]**2 + r1[1]**2 + r1[2]**2)**(0.5)
            visibility = dot_r1r2 - mod_r1 * self._R_E * math.sin(math.radians(self.station.elevation_min))

            if visibility > 0 and not comm_flag:
                start_comm_session = pos_sat
                comm_flag = True
            elif visibility < 0 and comm_flag:
                end_comm_session = pos_sat
                session_times.append((start_comm_session, end_comm_session))
                start_comm_session, end_comm_session = None, None
                comm_flag = False

        if comm_flag and not end_comm_session:
            session_times.append((start_comm_session, pos_sat))

        return session_times

    def _predict_one_comm_session(self, comm_index: int) -> None:
        '''Define communication parameters (azimuth, elevation, uplink and downlink
        frequencies) every second in duration of the one communcation session.'''
        start_index = (self.satellite.pos_ecef.index(self.session_times[comm_index][0]) -
                       self._PREDICT_SESSION_DELAY)
        if start_index < 0:
            start_index = 0
        end_index = (self.satellite.pos_ecef.index(self.session_times[comm_index][1]) +
                     self._PREDICT_SESSION_DELAY)
        if end_index > len(self.satellite.pos_ecef) - 1:
            end_index = len(self.satellite.pos_ecef) - 1

        for i in range(start_index + 1, end_index + 1):
            if (self.session_params[comm_index].start_session_dt >= self.satellite.pos_ecef[i].dt or
                self.session_params[comm_index].end_session_dt <= self.satellite.pos_ecef[i].dt):

                azimuth, elevation = self.calculate_azimuth_elevation(self.satellite.pos_ecef[i].xyz,
                                                                        self.station.ecef_position)
                self.comm_data_for_all_sat_prediction[self.satellite.pos_ecef[i].dt] = CommParams(elevation=elevation,
                                                                           azimuth=azimuth)

            if self.satellite.downlink_freq:
                uplink, downlink = self.calculate_uplink_downlink(self.satellite.pos_ecef[i - 1].xyz,
                                                                self.satellite.pos_ecef[i].xyz,
                                                                self.station.ecef_position)
                self.comm_data_for_all_sat_prediction[self.satellite.pos_ecef[i].dt].uplink = uplink
                self.comm_data_for_all_sat_prediction[self.satellite.pos_ecef[i].dt].downlink = downlink

    def define_session_params(self):
        '''Define main parameters of session which are described in SessionParams dataclass'''
        self.session_times = self._calculate_comm_session_times_for_prediction_period()
        for start_session, end_session in self.session_times:
            session = SessionParams(start_session.dt, end_session.dt)

            start_index = self.satellite.pos_ecef.index(start_session)
            end_index = self.satellite.pos_ecef.index(end_session)

            local_comm_data: dict[datetime, CommParams] = {}
            azimuth_prev: float = None
            for i in range(start_index, end_index):
                azimuth, elevation = self.calculate_azimuth_elevation(self.satellite.pos_ecef[i].xyz,
                                                                      self.station.ecef_position)
                local_comm_data[self.satellite.pos_ecef[i].dt] = CommParams(elevation=elevation,
                                                                            azimuth=azimuth)

                if azimuth_prev:
                    if abs(azimuth_prev - azimuth) > 330:
                        session.zero_crossing_azimuth_flag = True
                azimuth_prev = azimuth

                if i == start_index:
                    session.start_azimuth = azimuth
                    session.start_elevation = elevation
                elif i == end_index - 1:
                    session.end_azimuth = azimuth
                    session.end_elevation = elevation

            session.max_session_dt = max(local_comm_data, key=lambda dt: local_comm_data[dt].elevation)
            session.max_azimuth = local_comm_data[session.max_session_dt].azimuth
            session.max_elevation = local_comm_data[session.max_session_dt].elevation
            self.session_params.append(session)

            self.comm_data_for_all_sat_prediction.update(local_comm_data)

    def predict_comm_session_by_index(self, comm_index: int) -> None:
        '''Define communication parameters (azimuth, elevation, uplink and downlink
        frequencies) every second in duration of communcation sessions.
        If comm_index=-1 definition will be complete for all sessions.'''
        if comm_index != -1:
            self._predict_one_comm_session(comm_index)
        else:
            [self._predict_one_comm_session(comm_index) for comm_index in len(self.session_params)]

    def calculate_uplink_downlink(self, xyz_ecef_sat1: list[float],
                                  xyz_ecef_sat2: list[float],
                                  xyz_ecef_station: list[float]) -> list[float, float]:
        '''Caclulates uplink and downlink frequencies using two nearest
        positions of the satellite.'''
        r1: float = ((xyz_ecef_sat1[0] - xyz_ecef_station[0])**2 +
                     (xyz_ecef_sat1[1] - xyz_ecef_station[1])**2 +
                     (xyz_ecef_sat1[2] - xyz_ecef_station[2])**2)**(0.5)
        r2: float = ((xyz_ecef_sat2[0] - xyz_ecef_station[0])**2 +
                     (xyz_ecef_sat2[1] - xyz_ecef_station[1])**2 +
                     (xyz_ecef_sat2[2] - xyz_ecef_station[2])**2)**(0.5)
        v = r2 - r1

        uplink = self.satellite.uplink_freq / (1 - v / self._c)
        downlink = self.satellite.downlink_freq / (1 + v / self._c)

        return [uplink, downlink]

    def calculate_azimuth_elevation(self, xyz_ecef_sat: list[float],
                                    xyz_ecef_station: list[float]) -> list[float, float]:
        '''Calculates azimuth and elevation angle between the satellite
        and the ground station. Return azimuth and elevation in DEGREES!'''
        # Azimuth calculation
        lonlatalt_sat = self._transform_ecef_to_geodetic(xyz_ecef_sat)
        delta = lonlatalt_sat[0] - self.station.geo_position[0]
        Az: float = math.atan2(math.sin(delta) * math.cos(lonlatalt_sat[1]),
                               math.cos(self.station.geo_position[1]) * math.sin(lonlatalt_sat[1]) -
                               math.sin(self.station.geo_position[1]) * math.cos(lonlatalt_sat[1]) * math.cos(delta))
        if Az < 0:
            Az += 2 * math.pi

        # Elevation angle calculation
        r1 = [xyz_ecef_sat[0] - xyz_ecef_station[0], xyz_ecef_sat[1] - xyz_ecef_station[1], xyz_ecef_sat[2] - xyz_ecef_station[2]]
        r2 = [xyz_ecef_station[0], xyz_ecef_station[1], xyz_ecef_station[2]]
        dot_r1r2 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]
        mod_r1 = (r1[0]**2 + r1[1]**2 + r1[2]**2)**(0.5)
        mod_r2 = (r2[0]**2 + r2[1]**2 + r2[2]**2)**(0.5)
        sin_El = dot_r1r2 / (mod_r1*mod_r2)
        El: float = math.asin(sin_El)

        return [math.degrees(Az), math.degrees(El)]


class Orbitron:
    def __init__(self):
        self.stations: dict[str, GroundStation] = {}
        self.satellites: dict[str, Satellite] = {}
        self.comm_sessions: dict[str, SatelliteStationComm] = {}

    def setup_ground_station(self, longitude: float, latitude: float,
                             altitude: float, min_elevation: float, name='default') -> None:
        '''Setups ground station to NewOrbitron.'''
        self.stations[name] = GroundStation((longitude, latitude, altitude), min_elevation, name)
        print(f'{name} ground station {longitude=} deg, {latitude=} deg and {altitude=} m is defined.')

    def setup_satellite(self, norad_id: int, uplink: float = None, downlink: float = None) -> None:
        '''Setups satellites to NewOrbitron by norad_id.'''
        self.satellites[norad_id] = Satellite(norad_id, uplink, downlink)
        print(f'{norad_id} satellite is defined.')

    def setup_comm_session(self, norad_id: int, station_name: str = 'default') -> None:
        '''Setups communication session between ground station and satellite which have setup.'''
        if norad_id not in self.satellites:
            raise NewOrbitronSetupError(f'Orbitron has not setup for {norad_id} satellite.')
        elif station_name not in self.stations:
            raise NewOrbitronSetupError(f'Orbitron has not setup for station with name {station_name}.')

        self.comm_sessions[norad_id] = SatelliteStationComm(self.satellites[norad_id], self.stations[station_name])
        print(f'Communication session between ground station {station_name} and satellite {norad_id} is defined.')

    def predict_session_params(self, norad_id: int,
                               start_prediction: datetime = datetime.utcnow().replace(microsecond=0),
                               time_prediction: int = 86400, step_prediction: int = 1) -> None:
        '''Define all communication sessions and their parameters for predicted period.'''
        if norad_id not in self.comm_sessions:
            raise NewOrbitronSetupError(f'Orbitron has noone station in communication with {norad_id} satellite.')

        self.satellites[norad_id].predict_cm(start_prediction, time_prediction, step_prediction)
        self.comm_sessions[norad_id].define_session_params()
        print(f'{len(self.comm_sessions[norad_id].session_params)} sessions for {norad_id} satellite were defined.')

    def predict_comm_session(self, norad_id: int, comm_index: int = 0) -> None:
        '''Calculate parameters for communication session(s):
        comm_index - index of required communication session.
        By default com_index is zero, i.e. first communication session.
        You can use comm_index=-1 to predict all possible communcation sessions.'''
        if norad_id not in self.comm_sessions:
            raise NewOrbitronSetupError(f'Orbitron has noone station in communication with {norad_id} satellite.')
        elif not self.comm_sessions[norad_id].session_params or comm_index > len(self.comm_sessions[norad_id].session_params):
            raise NewOrbitronIndexError(f'Noone communication session is not defined or'
                                        'communication session with {comm_index} index does not exist.')
        self.comm_sessions[norad_id].predict_comm_session_by_index(comm_index)
        if comm_index == -1:
            print(f'Parameters for {len(self.comm_sessions[norad_id].session_params)} communication sessions.'
                  'for {norad_id} satellite were defined.')
        else:
            print(f'Parameters for communication session #{comm_index} were defined.')

    def update_tles(self, norad_ids: list[int]) -> None:
        '''Updates TLE files for required in norad_ids satellites by SpaceTrack API.'''
        for norad_id in self.satellites:
            if norad_id in norad_ids:
                self.satellites[norad_id].update_tle()

    def get_azimuth_elevation(self, norad_id: int) -> list[datetime | float | None]:
        '''Gets azimuth and elevation values for required satellite at current datetime
        if existed else returns None.'''
        if norad_id not in self.comm_sessions:
            raise NewOrbitronSetupError(f'Orbitron has noone station in communication with {norad_id} satellite.')
        elif not self.comm_sessions[norad_id].comm_data_for_all_sat_prediction:
            raise NewOrbitronDataError(f"Orbitron hasn't predicted data for {norad_id} satellite.")

        try:
            dt_cur = datetime.utcnow().replace(microsecond=0)
            point: CommParams = self.comm_sessions[norad_id].comm_data_for_all_sat_prediction[dt_cur]
            return [dt_cur.isoformat(), point.azimuth, point.elevation]
        except:
            return [dt_cur.isoformat(), None, None]

    def get_frequencies(self, norad_id: int) -> list[datetime | float | None]:
        '''Gets uplink and downlink frequencies calculated with Doppler shift for
        required satellite at current datetime if existed else returns None.'''
        if norad_id not in self.comm_sessions:
            raise NewOrbitronSetupError(f'Orbitron has noone station in communication with {norad_id} satellite.')
        elif not self.comm_sessions[norad_id].comm_data_for_all_sat_prediction:
            raise NewOrbitronDataError(f"Orbitron hasn't predicted data for {norad_id} satellite.")

        try:
            dt_cur = datetime.utcnow().replace(microsecond=0)
            point: CommParams = self.comm_sessions[norad_id].comm_data_for_all_sat_prediction[dt_cur]
            return [dt_cur.isoformat(), point.uplink, point.downlink]
        except:
            return [dt_cur.isoformat(), None, None]

    def get_data(self, norad_id: int) -> list[datetime, float | None]:
        '''Gets azimuth, elevation, uplink and downlink frequencies calculated
        with Dopper shift for required satellite at current datetime if existed
        else returns None.'''
        if norad_id not in self.comm_sessions:
            raise NewOrbitronSetupError(f'Orbitron has noone station in communication with {norad_id} satellite.')
        elif not self.comm_sessions[norad_id].comm_data_for_all_sat_prediction:
            raise NewOrbitronDataError(f"Orbitron hasn't predicted data for {norad_id} satellite.")

        dt_cur: datetime = datetime.utcnow().replace(microsecond=0)
        try:
            point: CommParams = self.comm_sessions[norad_id].comm_data_for_all_sat_prediction[dt_cur]
            return [dt_cur.isoformat(), point.azimuth, point.elevation, point.uplink, point.downlink]
        except:
            return [dt_cur.isoformat(), None, None, None, None]

    def get_comm_sessions(self, norad_id: int) -> list[CommParams]:
        '''Gets communication sessions parameters, which is described in CommParams,
        for required satellite if predicted and defined.'''
        if norad_id not in self.comm_sessions:
            raise NewOrbitronSetupError(f'Orbitron has noone station in communication with {norad_id} satellite.')
        elif not self.comm_sessions[norad_id].session_params:
            raise NewOrbitronDataError(f"Orbitron hasn't predicted communication sessions for {norad_id} satellite.")

        return self.comm_sessions[norad_id].session_params

    def show_satellite_radar(self, norad_id: int, comm_index: int = 0) -> None:
        '''Run radar for communication session in chosen SatelliteStationComm by norad_id.
        By default used nearest communication session, i.e. comm_index=0.'''
        if norad_id not in self.comm_sessions:
            raise NewOrbitronSetupError(f'Orbitron has noone station in communication with {norad_id} satellite.')
        elif not self.comm_sessions[norad_id].session_params or comm_index > len(self.comm_sessions[norad_id].session_params):
            raise NewOrbitronIndexError(f'Noone communication session is not defined or'
                                        'communication session with {comm_index} index does not exist.')
        def run_radar(comm_session):
            app = QApplication(sys.argv)
            w = MainWindowRadar(comm_session, comm_index)
            app.exec_()

        threading.Thread(target=run_radar, args=(self.comm_sessions[norad_id], )).start()

    @staticmethod
    def sync_windows_time() -> None:    # Run VScode as amdinistrator
        '''Synchronize Windows datetime with datetime from NTP server.
        To use this function is necessary to run python with Administrator rights.'''
        ntp_client = NTPClient()     # Network Time Protocol
        cur_timestamp: float = ntp_client.request('ntp0.ntp-servers.net').tx_time
        cur_dt = datetime.fromtimestamp(cur_timestamp, timezone.utc)
        win32api.SetSystemTime(cur_dt.year,
                            cur_dt.month,
                            cur_dt.isocalendar()[2],         # Day of week number
                            cur_dt.day,
                            cur_dt.hour,
                            cur_dt.minute,
                            cur_dt.second,
                            int(cur_dt.microsecond / 1000))  # Millseconds

    @staticmethod
    def __update_all_tles__(norad_ids: list[int], token: dict[Literal['identity', 'password'], str] =
                            {'identity': 'godlike200040@gmail.com', 'password': 'SamaraUniversity2022'},
                            tle_data_folder: str ='tle') -> None:
        '''Downloads all required TLE files by SpaceTrack API.'''
        tle_data_dir: str = os.path.join(os.path.dirname(__file__), tle_data_folder)
        if not os.path.exists(tle_data_dir):
            os.mkdir(tle_data_dir)

        st: SpaceTrackClient = SpaceTrackClient(identity=token.get('identity'), password=token.get('password'))
        for norad_id in tqdm(norad_ids, desc='Downloading TLE files', colour='green'):
            tle: str = st.tle_latest(norad_cat_id=norad_id, orderby='epoch desc', limit=1, format='3le')
            if not tle:
                tqdm.write(f'No data for the {norad_id} satellite!')
                continue

            tle_info: str = tle.split('\n')[1]
            epoch_year: int = int(tle_info[18:20])
            epoch_day: int = int(tle_info[20:23])
            if epoch_year <= 50:
                epoch_year += 2000
            else:
                epoch_year += 1900

            epoch: datetime = datetime(epoch_year, 1, 1) + timedelta(days=epoch_day - 1)
            tle_file_name: str = f'{norad_id}__{str(epoch.date())}.tle'
            with open(os.path.join(tle_data_dir, tle_file_name), 'w', encoding='utf-8') as tle_file:
                tle_file.write(tle)

    @staticmethod
    def __delete_all_tles__(tle_data_folder: str = 'tle'):
        '''Deletes all existed TLE files in folder with TLE files.'''
        if not os.path.exists(tle_data_folder):
            print('Folder with tle files is not found.')

        tle_data_dir: str = os.path.join(os.path.dirname(__file__), tle_data_folder)
        files_lst: list = os.listdir(tle_data_dir)
        for file in files_lst:
            if '.tle' in file or '.3le' in file:
                os.remove(os.path.join(tle_data_dir, file))

    @staticmethod
    def logging_data(norad_id: int, uplink: float, downlink: float,
                     lon: float, lat: float, alt: float, min_el: float,
                     start_dt: datetime = datetime.now().replace(microsecond=0),
                     time_prediction: int = 86400, step_prediction: int = 1):
        '''Defines communication parameters for required satellite and
        ground station and logs it to csv file.'''
        station: GroundStation = GroundStation((lon, lat, alt), min_el)

        satellite: Satellite = Satellite(norad_id, uplink, downlink)
        satellite.update_tle()
        satellite.predict_cm(start_dt=start_dt, time_prediction=time_prediction, step_prediction=step_prediction)

        sessions: SatelliteStationComm = SatelliteStationComm(satellite, station)
        sessions.define_session_params()
        sessions.predict_comm_session(0)

        if not os.path.exists('LogData'):
            os.mkdir('LogData')

        with open('LogData\\' + str(norad_id) + '__new_orbLog.csv', 'w', encoding='utf-8', newline='') as data_file:
            writer = csv.writer(data_file)
            for dt, data in sorted(sessions.comm_data_for_all_sat_prediction.items()):
                dt_str: str = dt.strftime('%c')
                azimuth: int = int(data.azimuth)
                elevation: int = int(data.elevation) if data.elevation > 0 else 0
                uplink: int = int(data.uplink)
                downlink: int = int(data.downlink)
                writer.writerow([dt_str, azimuth, elevation, uplink, downlink])


if __name__ == '__main__':
    # data = Orbitron.logging_data(norad_id = 24793,
    #                              uplink = 437399600,
    #                              downlink = 437399600,
    #                              lon = 50.1776,
    #                              lat = 53.2120,
    #                              alt = 137,
    #                              min_el = 0,
    #                              start_dt = datetime.utcnow().replace(microsecond=0),
    #                              time_prediction = 600,
    #                              step_prediction = 1)

    orbitron = Orbitron()
    orbitron.setup_ground_station(50.17763, 53.21204, 137, 0, 'Samara')

    orbitron.setup_satellite(24793, 437399600, 437399600)
    orbitron.setup_comm_session(24793, 'Samara')
    orbitron.update_tles([24793])
    orbitron.predict_session_params(24793)
    orbitron.predict_comm_session(24793)

    # orbitron.sync_windows_time()

    orbitron.show_satellite_radar(24793)

    for i in range(10):
        time.sleep(1)
        print(orbitron.get_data(24793))

    pass