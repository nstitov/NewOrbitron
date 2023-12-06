import time

from orbitron_tcp import OrbitronTcpClient

norad_id = 57173
stations = [{"longitude": 50.17763, "latitude":  53.21204, "altitude":  137, "elevation":  0, "name": 'Samara'},
            {"longitude": 0, "latitude":  0, "altitude":  0, "elevation":  0, "name":  'zero'}]
satellites = [{"norad_id": 43927, "uplink": 437398600, "downlink": 437398600},
              {"norad_id": 43928, "uplink": 437398600, "downlink": 437398600},
              {"norad_id": 57173, "uplink": 437398600, "downlink": 437398600}]

with OrbitronTcpClient() as client:
    client.setup_satellites(satellites)
    # oritron_tcp_client.sync_windows_time()
    client.setup_ground_stations(stations)
    client.setup_satellites(satellites)
    client.setup_comm_session(norad_id, 'Samara')
    client.update_tles([norad_id])
    client.predict_session_params(norad_id)
    client.predict_comm_session(norad_id)
    client.show_satellite_radar(norad_id)
    comm_sessions = client.get_comm_sessions(norad_id)
    for i in range(10):
        geod_position = client.get_azimuth_elevation(norad_id)
        print(f'{geod_position["dt"]}: azimith {geod_position["azimuth"]}, '
                f'elevation: {geod_position["elevation"]}')

        freqs = client.get_frequencies(norad_id)
        print(f'{freqs["dt"]}: downlink {freqs["downlink"]}, uplink {freqs["uplink"]}')

        data = client.get_data(norad_id)
        print(f'{data["dt"]}: azimith {data["azimuth"]}, elevation: {data["elevation"]}, '
                f'downlink {data["downlink"]}, uplink {data["uplink"]}')
        time.sleep(1)

    print('All functions is completed!')