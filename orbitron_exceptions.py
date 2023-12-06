class NewOrbitronSetupError(Exception):
    '''Program can't find required satellite or station'''

class NewOrbitronIndexError(Exception):
    '''Program can't use this index.'''

class NewOrbitronDataError(Exception):
    '''This data aren't defined yet.'''