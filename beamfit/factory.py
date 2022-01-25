registered_objects = {}


def register(objtype: str, name: str, createfun: callable):
    """Registers the creation function with the given name and an object type"""
    if objtype not in registered_objects:
        registered_objects[objtype] = {}
    if name in registered_objects[objtype]:
        raise ValueError(f"Name '{name}' of type '{objtype}' is already registered")
    registered_objects[objtype][name] = createfun


def create(objtype: str, name: str, **kwargs):
    """Returns an object created with the provided keyword arguments"""
    return registered_objects[objtype][name](**kwargs)


def unregister(objtype: str, name: str):
    registered_objects[objtype].pop(name)


def get_names(objtype: str):
    return list(registered_objects[objtype].keys())
