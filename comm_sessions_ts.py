from datetime import datetime, timedelta
from typing import Any
from environs import Env

import requests
import win32com.client

from operators_manager_tcp import OperatorsmanagerTcpClient
from orbitron_tcp import OrbitronTcpClient

HOST = "127.0.0.1"
time_offset = 4
start_time_delay = -2
stop_time_delay = 1

output_file_name = 'time_sessions.txt'
start_script_name = 'C:\\Users\\godli\\OneDrive\\Документы\\VolgaSpace\\new-orbitron\\start_test.bat'
stop_script_name = 'C:\\Users\\godli\\OneDrive\\Документы\\VolgaSpace\\new-orbitron\\stop_test.bat'

satellite_norad_id = 57173
satellite_uplink = 437398600
satellite_downlink = 437398600

samara_station_lon = 50.17763
samara_station_lat = 53.21204
samara_station_alt = 137
samara_station_el = 0

env = Env()
env.read_env()
BOT_TOKEN = env('BOT_TOKEN')
CHAT_ID = env('CHAT_ID')

def create_task(script_name: str, trigger_times: list[datetime], task_name: str, task_folder: str = 'ЦУП'):
    '''Sets tasks for start and stop communication scripts in Windows Task Scheduler.'''
    scheduler = win32com.client.Dispatch('Schedule.Service')
    scheduler.Connect()
    root_folder = scheduler.GetFolder(f'\\{task_folder}')
    task = scheduler.NewTask(0)
    for trigger_time in trigger_times:
        trigger = task.Triggers.Create(1)
        trigger.StartBoundary = trigger_time.isoformat()
    action = task.Actions.Create(0)
    action.ID = 'TRIGGER BATCH'
    action.Path = 'cmd.exe'
    action.Arguments = f'/c start "" {script_name}'
    # Register tasks
    task.RegistrationInfo.Description = task_name
    task.Settings.Enabled = True
    task.Settings.StopIfGoingOnBatteries = False
    # If task already exists, it will be updated
    TASK_CREATE_OR_UPDATE = 6
    TASK_LOGON_NONE = 0
    root_folder.RegisterTaskDefinition(
        task_name,  # Task name
        task,
        TASK_CREATE_OR_UPDATE,
        '',  # No user
        '',  # No password
        TASK_LOGON_NONE
    )
    print(f'В планировщик Windows добавлена задача {task_name} с {len(trigger_times)} триггер-ом(ами).')


def write_session_times_txt(start_dts: list[datetime], stop_dts: list[datetime],
                            filename: str = 'session_times.txt') -> None:
    '''Write start and end datetimes for communication sessions.'''
    with open(filename, 'w', encoding='utf-8') as f:
        for index, (start_dt, stop_dt) in enumerate(zip(start_dts, stop_dts)):
            f.write(f'{index + 1} сеанс связи: {start_dt.isoformat()} - {stop_dt.isoformat()};\n')
    print(f'В файл {filename} записана информация о {index} сеанс-е(ах) связи.')


def write_max_elevation_sessions_info_to_telegram(sessions: list[dict[str, Any]],
                                                  evening_operator: str,
                                                  morning_operator: str,
                                                  BOT_TOKEN: str,
                                                  chat_id: int,
                                                  time_offset: int,
                                                  time_delay: int) -> None:
    '''Write information about best evening and best morning communication sessions
    depend on elevation angle.'''
    evening_sessions, morning_sessions = [], []
    for session in sessions:
        session_dt = (datetime.fromisoformat(session['start_session_dt']) +
                      timedelta(hours=time_offset, minutes=time_delay))
        if session_dt.hour > 15:
            evening_sessions.append(session)
        else:
            morning_sessions.append(session)

    best_evening_session = max(evening_sessions, key=lambda session: session['max_elevation'])
    best_morning_session = max(morning_sessions, key=lambda session: session['max_elevation'])

    intro_text = 'Расписание на два сеанса:\n'
    start_evening_session = (datetime.fromisoformat(best_evening_session['start_session_dt']) +
                             timedelta(hours=time_offset, minutes=time_delay)).strftime('%H:%M')
    end_evening_session = (datetime.fromisoformat(best_evening_session['end_session_dt']) +
                             timedelta(hours=time_offset)).strftime('%H:%M')
    evening_session_text = f'1. Время начала: {start_evening_session}, время окончания: {end_evening_session}, оператор: {evening_operator};\n'

    start_morning_session = (datetime.fromisoformat(best_morning_session['start_session_dt']) +
                             timedelta(hours=time_offset, minutes=time_delay)).strftime('%H:%M')
    end_morning_session = (datetime.fromisoformat(best_morning_session['end_session_dt']) +
                             timedelta(hours=time_offset)).strftime('%H:%M')
    morning_session_text = f'2. Время начала: {start_morning_session}, время окончания: {end_morning_session}, оператор: {morning_operator}.'

    text = intro_text + evening_session_text + morning_session_text
    requests.get(f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={chat_id}&text={text}')
    print('Информация в телеграм канал отправлена.')


with OrbitronTcpClient(PORT=32768) as orbitron_client:
    stations = [{"longitude": samara_station_lon,
                 "latitude":  samara_station_lat,
                 "altitude":  samara_station_alt,
                 "elevation":  samara_station_el,
                 "name": 'Samara'}]
    satellites = [{"norad_id": satellite_norad_id,
                   "uplink": satellite_uplink,
                   "downlink": satellite_downlink}]

    orbitron_client.setup_ground_stations(stations)
    orbitron_client.setup_satellites(satellites)
    orbitron_client.setup_comm_session(satellite_norad_id, 'Samara')
    orbitron_client.update_tles([satellite_norad_id])
    orbitron_client.predict_session_params(satellite_norad_id)
    sessions = orbitron_client.get_comm_sessions(satellite_norad_id)

start_sessions = [datetime.fromisoformat(session['start_session_dt']) +
                                         timedelta(hours=time_offset, minutes=start_time_delay) for session in sessions]
create_task(script_name=start_script_name,
            trigger_times=start_sessions,
            task_folder='Start software')

stop_sessions = [datetime.fromisoformat(session['end_session_dt']) +
                                       timedelta(hours=time_offset, minutes=stop_time_delay) for session in sessions]
create_task(script_name=start_script_name,
            trigger_times=stop_sessions,
            task_folder='End software')

write_session_times_txt(start_sessions, stop_sessions, output_file_name)

with OperatorsmanagerTcpClient(PORT=32760) as operators_client:
    operators = operators_client.get_two_next_operators()
    print(f'Вечерний оператор - {operators["evening_operator"]};',
          f'утренний оператор - {operators["morning_operator"]}.', sep='\n')

write_max_elevation_sessions_info_to_telegram(sessions=sessions,
                                              evening_operator=operators['evening_operator'],
                                              morning_operator=operators['morning_operator'],
                                              BOT_TOKEN=BOT_TOKEN,
                                              chat_id=CHAT_ID,
                                              time_offset=time_offset,
                                              time_delay=start_time_delay)
