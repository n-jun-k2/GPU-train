class Const(object):

    ERROR_MESSAGE_TEMPLATE = "Can't rebind const (%s)"

    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError(self.ERROR_MESSAGE_TEMPLATE % name)
        self.__dict__[name] = value