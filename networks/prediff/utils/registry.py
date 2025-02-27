# Licensed to the GluonNLP team under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Create a registry."""

from typing import Optional, List
import json
from json import JSONDecodeError


class Registry:
    """Create the registry that will map name to object. This facilitates the users to create
    custom registry.

    Parameters
    ----------
    name
        The name of the registry

    Examples
    --------

    >>> from prediff.utils.registry import Registry
    >>> # Create a registry
    >>> MODEL_REGISTRY = Registry('MODEL')
    >>>
    >>> # To register a class/function with decorator
    >>> @MODEL_REGISTRY.register()
...     class MyModel:
...         pass
    >>> @MODEL_REGISTRY.register()
...     def my_model():
...         return
    >>>
    >>> # To register a class object with decorator and provide nickname:
    >>> @MODEL_REGISTRY.register('test_class')
...     class MyModelWithNickName:
...         pass
    >>> @MODEL_REGISTRY.register('test_function')
...     def my_model_with_nick_name():
...         return
    >>>
    >>> # To register a class/function object by function call
...     class MyModel2:
...         pass
    >>> MODEL_REGISTRY.register(MyModel2)
    >>> # To register with a given name
    >>> MODEL_REGISTRY.register('my_model2', MyModel2)
    >>> # To list all the registered objects:
    >>> MODEL_REGISTRY.list_keys()

['MyModel', 'my_model', 'test_class', 'test_function', 'MyModel2', 'my_model2']

    >>> # To get the registered object/class
    >>> MODEL_REGISTRY.get('test_class')

__main__.MyModelWithNickName

    """

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: dict[str, object] = dict()

    def _do_register(self, name: str, obj: object) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, *args):
        """
        Register the given object under either the nickname or `obj.__name__`. It can be used as
         either a decorator or not. See docstring of this class for usage.
        """
        if len(args) == 2:
            # Register an object with nick name by function call
            nickname, obj = args
            self._do_register(nickname, obj)
        elif len(args) == 1:
            if isinstance(args[0], str):
                # Register an object with nick name by decorator
                nickname = args[0]
                def deco(func_or_class: object) -> object:
                    self._do_register(nickname, func_or_class)
                    return func_or_class
                return deco
            else:
                # Register an object by function call
                self._do_register(args[0].__name__, args[0])
        elif len(args) == 0:
            # Register an object by decorator
            def deco(func_or_class: object) -> object:
                self._do_register(func_or_class.__name__, func_or_class)
                return func_or_class
            return deco
        else:
            raise ValueError('Do not support the usage!')

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret

    def list_keys(self) -> List:
        return list(self._obj_map.keys())

    def __repr__(self) -> str:
        s = '{name}(keys={keys})'.format(name=self._name,
                                         keys=self.list_keys())
        return s

    def create(self, name: str, *args, **kwargs) -> object:
        """Create the class object with the given args and kwargs

        Parameters
        ----------
        name
            The name in the registry
        args
        kwargs

        Returns
        -------
        ret
            The created object
        """
        obj = self.get(name)
        try:
            return obj(*args, **kwargs)
        except Exception as exp:
            print('Cannot create name="{}" --> {} with the provided arguments!\n'
                  '   args={},\n'
                  '   kwargs={},\n'
                  .format(name, obj, args, kwargs))
            raise exp

    def create_with_json(self, name: str, json_str: str):
        """

        Parameters
        ----------
        name
        json_str

        Returns
        -------

        """
        try:
            args = json.loads(json_str)
        except JSONDecodeError:
            raise ValueError('Unable to decode the json string: json_str="{}"'
                             .format(json_str))
        if isinstance(args, (list, tuple)):
            return self.create(name, *args)
        elif isinstance(args, dict):
            return self.create(name, **args)
        else:
            raise NotImplementedError('The format of json string is not supported! We only support '
                                      'list/dict. json_str="{}".'
                                      .format(json_str))
