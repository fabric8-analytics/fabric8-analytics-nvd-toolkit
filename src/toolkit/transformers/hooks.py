"""This module contains the Hook class to handle pipeline hooks."""

import weakref


class Hook(object):
    """Convenient class for handling hooks.

    :param key: str, unique identifier of the hook
    :param func: function to be called by the hook

        The function can not modify any items fed by its arguments.
    :param default_kwargs: default `func` keyword argument values

        Example:
        def foo(x, verbose=False):
            if verbose:
                print('verbosity on')
            return x

        # init with default kwargs
        foo_hook = Hook('foo', foo, verbose=True)
        # and on the call
        foo_hook(x=None)  # prints 'verbosity on'

    :param reuse: whether to reuse (share) the Hook

    """

    __INSTANCES = weakref.WeakSet()

    def __init__(self, key: str, func, reuse=False, **default_kwargs):
        """Initialize hook."""
        if key in Hook.get_current_keys():
            if not reuse:
                raise ValueError("Hook with key `%s` already exists" % key)
            else:
                # TODO: share the existing hook instead of creating a new one
                pass

        # attr initialization
        self._key = str(key)
        self._func = func
        self._default_kwargs = default_kwargs

        # add the key to the class
        Hook.__INSTANCES.add(self)

    @property
    def key(self):
        """Get hook key."""
        return self._key

    @property
    def default_kwargs(self):
        """Get hook default keyword arguments."""
        return self._default_kwargs

    @default_kwargs.setter
    def default_kwargs(self, kwargs):
        self._default_kwargs = kwargs

    def __call__(self, *args, **kwargs):
        """Call the hooked function."""
        return self._func(*args, **kwargs)

    @classmethod
    def get_current_hooks(cls) -> list:
        """Return instances of this class."""
        return list(cls.__INSTANCES)

    @classmethod
    def get_current_keys(cls) -> set:
        """Return keys to the instances of this class."""
        return set([hook.key for hook in cls.__INSTANCES])

    @classmethod
    def clear_current_instances(cls):
        """Clean up the references held by the class.

        This function is not usually called by user, mainly used for tests
        where cleanup is needed.
        """
        cls.__INSTANCES.clear()
