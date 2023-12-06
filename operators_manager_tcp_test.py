from operators_manager_tcp import OperatorsmanagerTcpClient

with OperatorsmanagerTcpClient() as client:
    operators = client.get_two_next_operators()
    print(f'Evening operator - {operators["evening_operator"]};',
          f'morning operators - {operators["morning_operator"]}.', sep='\n')