def validate_iterable(iterable):
    '''Raises TypeError if input is not an iterable'''
    if not isinstance(iterable, (tuple, list, set)):
        raise TypeError(u'Received unsupported type "{}", expected iterable'.format(type(iterable)))
