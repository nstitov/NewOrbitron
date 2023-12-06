import json
import socket
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import Any


class TCPServer(ABC):
    '''An abstract class used to create TCP servers.'''
    _ThreadCounter: int = 0

    class ResponseType(IntEnum):
        NONE = 0
        CONFIGURE = 1
        PREDICT = 2
        TLE_UPDATE = 3
        SYNC = 4
        RADAR = 5
        GET_DATA = 6

    def __init__(self, HOST: str | int = '127.0.0.1', PORT: int = 32768, time_offset: int = 4):
        self._HOST = HOST
        self._PORT = PORT
        self._timezone = timezone(timedelta(hours=time_offset))
        sock: socket.socket = socket.socket()

        try:
            sock.bind((self._HOST, self._PORT))
        except socket.error as e:
            print(e)

        print(f'Server is listing on the port {self._PORT}...')
        sock.listen()

        while True:
            self.accept_connections(sock)

    def client_handler(self, connection: socket.socket) -> None:
        while True:
            data: bytes = connection.recv(2048)
            message: str = data.decode('utf-8')
            if message == 'CLOSE':
                break

            if message != None and message != '':
                resp: tuple = self.resolve_message(message)

                if resp[0] == self.ResponseType.GET_DATA:
                    data: str = json.dumps(resp[1]) + json.dumps(resp[0])
                    connection.sendall(data.encode('utf-8'))
                else:
                    connection.sendall(json.dumps(resp[0]).encode('utf-8'))

        connection.close()

    def accept_connections(self, sock: socket.socket) -> None:
        Client, address = sock.accept()
        self._ThreadCounter += 1
        print('Connected to: ' + address[0] + ':' + str(address[1]))
        threading.Thread(target=self.client_handler, args=(Client, ) ).start()

    def resolve_message(self, message) -> None | tuple[ResponseType]:
        msg: dict = json.loads(message)

        current_time = datetime.now(self._timezone)

        if 'request' in msg.keys():
            return self.handle_request_msg(msg, current_time)

    @abstractmethod
    def handle_request_msg(self, msg: dict, current_time: datetime) -> tuple[ResponseType, dict[str, Any]]:
        '''Processes the request massage depending on its body.'''
        print(current_time)


class TCPClient(ABC):
    '''An abstract class used to create TCP clients that can be used with context manager.'''
    def __init__(self, HOST: str | int = '127.0.0.1', PORT: int = 32768):
        self._HOST = HOST
        self._PORT = PORT

    def __enter__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self._HOST, self._PORT))
        time.sleep(1)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sock.sendall("CLOSE".encode('utf-8'))
        self.sock.close()
        if isinstance(exc_type, OSError):
            print(f'Ошибка в подключении к TCP серверу: {exc_value}')
            return True
        return False
