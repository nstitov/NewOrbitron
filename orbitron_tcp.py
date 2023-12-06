import json
import time
from datetime import datetime
from typing import Any, Literal

from new_orbitron import Orbitron
from TcpServerABC import TCPClient, TCPServer


class OrbitronTcpServer(TCPServer):
    def __init__(self, HOST: str | int = '127.0.0.1', PORT: int = 32768, time_offset: int = 4):
        self.orbitron = Orbitron()
        super().__init__(HOST, PORT, time_offset)

    def handle_request_msg(self, msg: dict, current_time: datetime) -> tuple[TCPServer.ResponseType, dict[str, Any]]:
        super().handle_request_msg(msg, current_time)

        if msg['request'] == "setup_stations":
            if 'body' in msg:
                for station in msg['body']:
                    self.orbitron.setup_ground_station(station['longitude'],
                                                       station['latitude'],
                                                       station['altitude'],
                                                       station.get('elevation', 0),
                                                       station.get('name', 'default'))
                return (self.ResponseType.CONFIGURE, )
            raise Exception('No body in setup_station request')

        elif msg['request'] == "setup_satellites":
            if 'body' in msg:
                for satellite in msg['body']:
                    self.orbitron.setup_satellite(satellite['norad_id'],
                                                  satellite.get('uplink', None),
                                                  satellite.get('downlink', None))
                return (self.ResponseType.CONFIGURE, )
            raise Exception('No body in setup_satellites request')

        elif msg['request'] == "setup_comm_session":
            if 'body' in msg:
                self.orbitron.setup_comm_session(msg['body']['norad_id'],
                                                 msg['body']['station_name'])
                return (self.ResponseType.CONFIGURE, )
            raise Exception('No body in setup_session request')

        elif msg['request'] == "predict_session_params":
            if 'body' in msg:
                self.orbitron.predict_session_params(msg['body']['norad_id'],
                                                     datetime.fromisoformat(msg['body']['start_prediction']),
                                                     msg['body']['time_prediction'],
                                                     msg['body']['step_prediction'])
                return (self.ResponseType.PREDICT, )
            raise Exception('No body in predict_session_params request')

        elif msg['request'] == "predict_comm_session":
            if 'body' in msg:
                self.orbitron.predict_comm_session(msg['body']['norad_id'],
                                                   msg['body']['comm_index'])
                return (self.ResponseType.PREDICT, )
            raise Exception('No body in predict_comm_session request')

        elif msg['request'] == "update_tles":
            if 'body' in msg:
                self.orbitron.update_tles(msg['body']['norad_ids'])
                return (self.ResponseType.TLE_UPDATE, )
            raise Exception("No body in update request")

        elif msg['request'] == "sync_windows_time":
            self.orbitron.sync_windows_time()
            return (self.ResponseType.SYNC, )

        elif msg['request'] == "get_azimuth_elevation":
            if 'body' in msg:
                data = self.orbitron.get_azimuth_elevation(msg['body']['norad_id'])
                if data[1]:
                    print(f'{data[0]}: az. {data[1]:03}, el. {data[2]:03}.')
                    return (self.ResponseType.GET_DATA, {"dt": data[0],
                                                         "azimuth": data[1],
                                                         "elevation": data[2]})
                else:
                    print(f'{data[0]}: no visible.')
                    return (self.ResponseType.GET_DATA, {"dt": data[0],
                                                         "azimuth": f'No visible',
                                                         "elevation": f'No visible'})

        elif msg['request'] == "get_frequencies":
            if 'body' in msg:
                data = self.orbitron.get_frequencies(msg['body']['norad_id'])
                if data[1]:
                    print(f'{data[0]}: upl. {int(data[1])}, dnl. {int(data[2])}.')
                    return (self.ResponseType.GET_DATA, {"dt": data[0],
                                                         "uplink": f'{int(data[1])}',
                                                         "downlink": f'{int(data[2])}'})
                else:
                    print(f'{data[0]}: no visible.')
                    return (self.ResponseType.GET_DATA, {"dt": data[0],
                                                         "uplink": 'No visible',
                                                         "downlink": 'No visible'})

        elif msg['request'] == "get_data":
            if 'body' in msg:
                data = self.orbitron.get_data(msg['body']['norad_id'])
                if data[1]:
                    if data[3]:
                        print(f'{data[0]}: az. {data[1]:03}, el. {data[2]:03}, upl. {int(data[3])}, dnl. {int(data[4])}')
                        return (self.ResponseType.GET_DATA, {"dt": data[0],
                                                             "azimuth": data[1],
                                                             "elevation": data[2],
                                                             "uplink": f'{int(data[3])}',
                                                             "downlink": f'{int(data[4])}'})
                    else:
                        print(f'{data[0]}: az. {data[1]:03}, el. {data[2]:03}, upl. no def., dnl. no def.')
                        return (self.ResponseType.GET_DATA, {"dt": data[0],
                                                             "azimuth": data[1],
                                                             "elevation": data[2],
                                                             "uplink": 'No visible',
                                                             "downlink": 'No visible'})
                else:
                    print(f'{data[0]}: no visible.')
                    return (self.ResponseType.GET_DATA, {"dt": data[0],
                                                         "azimuth": 'No visible',
                                                         "elevation": 'No visible',
                                                         "uplink": 'No visible',
                                                         "downlink": 'No visible'})
            raise Exception("No body in get data")

        elif msg['request'] == "get_comm_sessions":
            if 'body' in msg:
                sessions = self.orbitron.get_comm_sessions(msg['body']['norad_id'])
                js = []
                for session in sessions:
                    session_js = {'start_session_dt': session.start_session_dt.isoformat(),
                                  'start_elevation': session.start_elevation,
                                  'start_azimuth': session.start_azimuth,
                                  'start_sun_azimuth': session.start_sun_azimuth,
                                  'start_sun_elevation': session.start_sun_elevation,
                                  'end_session_dt': session.end_session_dt.isoformat(),
                                  'end_elevation': session.end_elevation,
                                  'end_azimuth': session.end_azimuth,
                                  'end_sun_azimuth': session.end_sun_azimuth,
                                  'end_sun_elevation': session.end_sun_elevation,
                                  'max_session_dt': session.max_session_dt.isoformat(),
                                  'max_elevation': session.max_elevation,
                                  'max_azimuth': session.max_azimuth,
                                  'max_sun_azimuth': session.max_sun_azimuth,
                                  'max_sun_elevation': session.max_sun_elevation,
                                  'zero_crossing_azimuth_flag': session.zero_crossing_azimuth_flag}
                    js.append(session_js)
                print(f'Got parameters for {len(js)} communication sessions.')
                return (self.ResponseType.GET_DATA, js)

        elif msg['request'] == "show_satellite_radar":
            if 'body' in msg:
                self.orbitron.show_satellite_radar(msg['body']['id'],
                                                   msg['body']['comm_index'])
                return (self.ResponseType.RADAR, )
            raise Exception("No body in radar")

        else:
            return (self.ResponseType.NONE, )


class OrbitronTcpClient(TCPClient):
    '''Total representation of NewOrbitron API commands
    to interactions with NewOrbitron by TCP server.
    Can be used with the context manager.'''
    def setup_ground_stations(self, stations: list[dict[str, float | int| str]]) -> OrbitronTcpServer.ResponseType:
        '''Sends command to NewOrbitron TCP server to setup ground stations.'''
        js = {"request": "setup_stations", "body": stations}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        resp = self.sock.recv(4).decode('utf-8')
        return resp

    def setup_satellites(self, satellites: list[dict[str, int]]) -> OrbitronTcpServer.ResponseType:
        '''Sends command to NewOrbitron TCP server to setup satellites.'''
        js = {"request": "setup_satellites",
            "body": satellites}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        resp = self.sock.recv(4).decode('utf-8')
        return resp

    def setup_comm_session(self, norad_id: int, station_name: str) -> OrbitronTcpServer.ResponseType:
        '''Sends command to NewOrbitron TCP server to setup SatelliteStationComm
        between the one satellite and the one ground station'''
        js = {"request": "setup_comm_session", "body": {"norad_id": norad_id,
                                                        "station_name": station_name}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        resp = self.sock.recv(4).decode('utf-8')
        return resp

    def predict_session_params(self, norad_id: int,
                               start_prediction: datetime = datetime.utcnow().replace(microsecond=0),
                               time_prediction: int = 86400,
                               step_prediction: int = 1) -> OrbitronTcpServer.ResponseType:
        '''Sends command to NewOrbitron TCP server to predict communication sessions
        parameters within one setuped SatelliteStationComm.'''
        js = {"request": "predict_session_params", "body": {"norad_id": norad_id,
                                                            "start_prediction": start_prediction.isoformat(),
                                                            "time_prediction": time_prediction,
                                                            "step_prediction": step_prediction}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        resp = self.sock.recv(4).decode('utf-8')
        return resp

    def predict_comm_session(self, norad_id: int, comm_index: int = 0) -> OrbitronTcpServer.ResponseType:
        '''Sends command to NewOrbitron TCP server to predict one or all communication
        sessions for the setuped satellite within one setuped SatelliteStationComm.
        Specify index of required session by comm_index parameters.
        If you want to predict all possible sessions within one SatelliteStationComm then
        enter comm_index=-1.'''
        js = {"request": "predict_comm_session", "body": {"norad_id": norad_id,
                                                          "comm_index": comm_index}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        resp = self.sock.recv(4).decode('utf-8')
        return resp

    def update_tles(self, norad_ids: list[int]) -> OrbitronTcpServer.ResponseType:
        '''Sends command to NewOrbitron TCP server to update required TLE files by
        SpaceTrack API client using satellite NORAD IDs.'''
        js =  {"request": "update_tles", "body": {"norad_ids": norad_ids}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        resp = self.sock.recv(4).decode('utf-8')
        return resp

    def sync_windows_time(self) -> OrbitronTcpServer.ResponseType:
        '''Sends command to NewOrbitron TCP server to synchronize
        local Windows time with time from NTP server. To use this function
        the python must be running with Administrator rights.'''
        js =  {"request": "sync"}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        resp = self.sock.recv(4).decode('utf-8')
        return resp

    def get_azimuth_elevation(self, norad_id: int) -> dict[Literal['dt', 'azimuth', 'elevation'], str]:
        '''Sends command to NewOrbitron TCP server to get azimuth and elevation
        values at current datetime.'''
        js = {"request": "get_azimuth_elevation", "body": {"norad_id": norad_id}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        time.sleep(0.1)
        data = self.sock.recv(2048).decode('utf-8')
        resp = data[-1]
        data = json.loads(data[:-1])
        return data

    def get_frequencies(self, norad_id: int) -> dict[Literal['dt', 'uplink', 'downlink'], str]:
        '''Sends command to NewOrbitron TCP server to get uplink and downlink
        frequencies at currant datetime.'''
        js = {"request": "get_frequencies", "body": {"norad_id": norad_id}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        time.sleep(0.1)
        data = self.sock.recv(2048).decode('utf-8')
        resp = data[-1]
        data = json.loads(data[:-1])
        return data

    def get_data(self, norad_id: int) -> dict[Literal['dt', 'azimuth', 'elevation', 'uplink','downlink'], str]:
        '''Sends command to NewOrbitron TCP server to get communication session
        data at currnt datetime: azimuth, elevation, uplink and downlink
        frequencies.'''
        js =  {"request": "get_data", "body": {"norad_id": norad_id}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        time.sleep(0.1)
        data = self.sock.recv(2048).decode('utf-8')
        resp = data[-1]
        data = json.loads(data[:-1])
        return data

    def get_comm_sessions(self, norad_id: int) -> list[dict[str, str | float | None | bool]]:
        '''Sends command to NewOtbitron TCP server to get parameters of
        predicted communication sessions for required satellite.'''
        js = {"request": "get_comm_sessions", "body": {"norad_id": norad_id}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        time.sleep(0.5)
        data = self.sock.recv(8192).decode('utf-8')
        resp = data[-1]
        data = json.loads(data[:-1])
        return data

    def show_satellite_radar(self, norad_id: int, comm_index: int = 0) -> OrbitronTcpServer.ResponseType:
        '''Sends command to NewOrbitron TCP server to show radar with satellite
        communication session.'''
        js =  {"request": "show_satellite_radar", "body": {"id": norad_id,
                                                        "comm_index": comm_index}}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        resp = self.sock.recv(4).decode('utf-8')
        time.sleep(0.5)
        return resp


if __name__ == '__main__':
    server = OrbitronTcpServer()
    pass