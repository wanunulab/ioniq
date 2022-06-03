from functools import wraps
# def singleton(orig_cls):
    
#     orig_new = orig_cls.__new__
#     instance = None

#     @wraps(orig_cls.__new__)
#     def __new__(cls, *args, **kwargs):
#         nonlocal instance
#         if instance is None:
#             instance = orig_new(cls, *args, **kwargs)
#         return instance
#     orig_cls.__new__ = __new__
#     return orig_cls

def singleton(cls):    
    """ 
    singleton decorator for a class
    usage: place @singleton before the class definintion
    taken from https://riptutorial.com/python/example/10954/create-singleton-class-with-a-decorator
    """
    instance = [None]
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if instance[0] is None:
            instance[0] = cls(*args, **kwargs)
        return instance[0]

    return wrapper