import math
import sys
from datetime import datetime, timedelta

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QMainWindow,
                             QScrollArea, QSizePolicy, QSpacerItem,
                             QVBoxLayout, QWidget)


matplotlib.use("Qt5Agg")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, comm_session, comm_index):
        radar = Figure()
        self.axes = radar.add_subplot(projection='polar')
        self.axes.set_theta_zero_location('N')
        self.axes.set_theta_direction(-1)
        self.axes.set_rmax(0)
        self.axes.set_rmin(90)
        self.axes.set_rticks([0, 15, 30, 45, 60, 75, 90])
        self.axes.set_thetagrids([0, 90, 180, 270])
        self.axes.grid(True)

        self.trace = self.axes.plot([], [], lw=1, color='g')[0]
        self.sat = self.axes.plot([], [], 'o', lw=3, ms=7, color='r')[0]

        azimuth, elevation = [], []
        start_session_dt = comm_session.session_params[comm_index].start_session_dt
        end_session_dt = comm_session.session_params[comm_index].end_session_dt
        while start_session_dt <= end_session_dt:
            point = comm_session.comm_data_for_all_sat_prediction[start_session_dt]
            azimuth.append(math.radians(point.azimuth))
            elevation.append(point.elevation)
            start_session_dt += timedelta(seconds=1)
        self.trace.set_data(azimuth, elevation)

        super().__init__(radar)


class MainWindowRadar(QMainWindow):
    def __init__(self, comm_session, comm_index):
        super().__init__()

        self.comm_session = comm_session
        self.comm_index = comm_index

        self.setWindowTitle(f'{comm_session.satellite.satellite_name}({comm_session.satellite.norad_id}) - {comm_session.station.name}')
        vertical_layout = QVBoxLayout()
        horizontal_layout = QHBoxLayout()
        widget = QWidget()
        sca_widget = QScrollArea()

        self.radar = MplCanvas(self.comm_session, comm_index)
        self.radar.axes.plot()

        self.comm_data = CommData(self.comm_session, self.comm_index)
        sca_widget.setWidget(CommSessions(self.comm_session))

        vertical_layout.addWidget(self.radar)
        vertical_layout.addWidget(self.comm_data)
        horizontal_layout.addLayout(vertical_layout)
        horizontal_layout.addWidget(sca_widget)

        widget.setLayout(horizontal_layout)
        self.setCentralWidget(widget)
        self.resize(1100, 580)
        self.setMaximumSize(1100, 580)
        self.setMinimumSize(1100, 580)
        self.show()

        self.timer = QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_sat_position)
        self.timer.start()

    def _form_data_for_chosen_session(self):
        one_session_data = {}
        start_session_dt = self.comm_session.session_params[self.comm_index].start_session_dt
        end_session_dt = self.comm_session.session_params[self.comm_index].end_session_dt
        while start_session_dt <= end_session_dt:
            one_session_data[start_session_dt] = self.comm_session.comm_data_for_all_sat_prediction[start_session_dt]

    def update_sat_position(self):
        dt = datetime.utcnow().replace(microsecond=0)
        try:
            point = self.comm_session.comm_data_for_all_sat_prediction[dt]
            azimuth, elevation, uplink, downlink = point.azimuth, point.elevation, point.uplink, point.downlink
        except KeyError:
            azimuth, elevation, uplink, downlink = None, None, None, None

        if azimuth:
            self.radar.sat.set_data(math.radians(azimuth), elevation)
        else:
            self.radar.sat.set_data(None, None)
        self.comm_data.update_comm_data(dt, azimuth, elevation, uplink, downlink)
        self.radar.draw()


class CommSessions(QWidget):

    class CommSession(QWidget):
        def __init__(self, session_time):
            super().__init__()
            uic.loadUi('ui/communication_sessions.ui', self)

            self.start_value.setText(str(session_time[0].dt))
            self.end_value.setText(str(session_time[1].dt))


    def __init__(self, comm_session):
        super().__init__()

        layout = QVBoxLayout()
        for session_time in comm_session.session_times:
            layout.addWidget(self.CommSession(session_time))
            verticalSpacer = QSpacerItem(10, 5, QSizePolicy.Maximum, QSizePolicy.Maximum)
            layout.addItem(verticalSpacer)

        self.setLayout(layout)
        self.resize(430, 500)


class CommData(QWidget):
    def __init__(self, comm_session, comm_index):
        super().__init__()
        uic.loadUi('ui/communication_data.ui', self)

        self.time_value.setText('None')
        self.azimuth_value.setText('No visible')
        self.elevation_value.setText('No visible')
        self.uplink_value.setText('No visible')
        self.downlink_value.setText('No visible')
        if comm_session.session_params[comm_index].zero_crossing_azimuth_flag:
            self.azimuth_flag_value.setText('True')
        else:
            self.azimuth_flag_value.setText('False')

        self.resize(500, 300)

    def update_comm_data(self, dt: datetime, azimuth: float, elevation: float, uplink: float, downlink: float):
        self.time_value.setText(dt.isoformat())
        if azimuth:
            self.azimuth_value.setText(f'{azimuth:.2f}')
            self.elevation_value.setText(f'{elevation:.2f}')
            if uplink:
                self.uplink_value.setText(str(int(uplink)))
                self.downlink_value.setText(str(int(downlink)))
            else:
                self.uplink_value.setText('Not defined')
                self.downlink_value.setText('Not defined')
        else:
            self.azimuth_value.setText('No visible')
            self.elevation_value.setText('No visible')
            self.uplink_value.setText('No visible')
            self.downlink_value.setText('No visible')


if __name__ == '__main__':

    from new_orbitron import Orbitron

    app = QApplication(sys.argv)
    w = MainWindowRadar()
    app.exec_()