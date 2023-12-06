import json
from datetime import date, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from TcpServerABC import TCPClient, TCPServer


class OperatorsManagerTcpServer(TCPServer):
    _operators_page_name = 'Операторы'

    def __init__(self, excel_filename: str, HOST: str | int = '127.0.0.1',
                 PORT: int = 32768, time_offset: int = 4):
        self.excel_filename = excel_filename
        super().__init__(HOST, PORT, time_offset)

    def _find_operator_fullname(self, shortname: str) -> str:
        '''Function to find the operator fullname by operator shortname.'''
        operators = pd.read_excel(self.excel_filename,
                                  sheet_name=self._operators_page_name,
                                  dtype=str)
        row, col = np.where(operators==shortname)
        if len(row) != 0:
            operator = operators.iat[row[0], col[0] + 1]
            return operator

    def _find_next_operator(self, req_date: date, morning: bool) -> str:
        '''Function to find operator shortname for morning or evening communication
        session, depending on the "morning" flag, using the required date.'''
        sheet = pd.read_excel(self.excel_filename, sheet_name=req_date.strftime('%m.%y'), dtype=str)
        row, col = np.where(sheet==req_date.strftime('%Y-%m-%d 00:00:00'))
        if len(row) != 0:
            if morning:
                operator = sheet.iat[row[0] + 1, col[0]]
            else:
                operator = sheet.iat[row[0] + 2, col[0]]

            operator_fullname = self._find_operator_fullname(operator)
            if operator_fullname:
                return operator_fullname
            return operator

    def _get_two_next_operators(self) -> tuple[str, str]:
        '''Find today evening and tomorrow morning operators shortnames in excel file.
        Should be started after the morning session'''
        tod_date = date.today()
        tom_date = date.today() + timedelta(days=1)
        evening_operator = self._find_next_operator(tod_date, morning=False)
        morning_operator = self._find_next_operator(tom_date, morning=True)
        return evening_operator, morning_operator

    def handle_request_msg(self, msg: dict, current_time: datetime) -> tuple[TCPServer.ResponseType, dict[str, Any]]:
        super().handle_request_msg(msg, current_time)

        if msg['request'] == 'get_two_next_operators':
            evening_operator, morning_operator = self._get_two_next_operators()
            return (self.ResponseType.GET_DATA, {'evening_operator': evening_operator,
                                                 'morning_operator': morning_operator})
        else:
            return (self.ResponseType.NONE, )


class OperatorsmanagerTcpClient(TCPClient):
    '''TCP client for interactions with OperatorsManager TCP server.'''
    def get_two_next_operators(self) -> dict[Literal['evening_operator', 'morning_operator'], str]:
        '''Sends command to OperatorsManager TCP server to get fullnames of
        two next operators in mission control center.'''
        js = {'request': 'get_two_next_operators'}
        self.sock.sendall(json.dumps(js).encode('utf-8'))
        data = self.sock.recv(256).decode('utf-8')
        resp = data[-1]
        operators = json.loads(data[:-1])
        return operators


if __name__ == '__main__':
    server = OperatorsManagerTcpServer(excel_filename='Расписание работ в ЦУП.xlsx', PORT=32760)
    pass