registered_objects = {}


def register(name: str, createfun: callable):
    """Registers the creation function with the given name and an object type"""
    if name in registered_objects:
        raise ValueError(f"Name '{name}' is already registered")
    registered_objects[name] = createfun


def create(name: str, **kwargs):
    """Returns an object created with the provided keyword arguments"""
    return registered_objects[name](**kwargs)


def unregister(name: str):
    registered_objects.pop(name)


def get_names():
    return list(registered_objects.keys())
