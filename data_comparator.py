import csv
import os.path


log_data_folder = 'LogData'

with (open(os.path.join(log_data_folder, '24793__orbLog.csv'), 'r', encoding='utf-8') as old_data_file,
      open(os.path.join(log_data_folder, '24793__new_orbLog.csv'), 'r', encoding='utf-8') as new_data_file):
    old_reader = csv.reader(old_data_file)
    new_reader = csv.reader(new_data_file)
    old_data, new_data = [], []
    for line in old_reader:
        old_data.append(line)
    for line in new_reader:
        new_data.append(line)

    for old_line, new_line in zip(old_data, new_data):
        assert abs(int(old_line[1]) - int(new_line[1])) <= 1, f'Отклонение расчёта азимута больше 1 градуса в момент местного времени {old_line[0]}'
        assert abs(int(old_line[2]) - int(new_line[2])) <= 1, f'Отклонение расчёта угла возвышения больше 1 градуса в момент местного времени {old_line[0]}'
        assert abs(int(old_line[3]) - int(new_line[3])) <= 3, f'Отклонение расчёта частоты передачи больше 3 Гц в момент местного времени {old_line[0]}'
        assert abs(int(old_line[4]) - int(new_line[4])) <= 3, f'Отклонение расчёта частоты приёма больше 3 Гц в момент местного времени {old_line[0]}'
    print('Погрешность моделирования допустима!')