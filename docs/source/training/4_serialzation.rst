.. _serialisation:
Serialisation Guide
=====================

***************
Serialization
***************

Sonosco provides it's own serialization method. It allows to save complete state of your
training. Including parameters and meta-parameters.

In order to enable the serialization follow those steps:

1. add the ``@sonosco.serialization.serializable`` decorator all the classes you wish to serialize.

2. Instead of using ``__init__()`` method put all the fields of your object on the class level (Type annotation are mandatory!)
   (Currently only primitives, lists of primitives, types, callables and other serializable objects are supported).
   If you wish to perform some custom initialization use ``__post_init__(self)`` method  (``__init__(self)`` is auto generated!)

::

    @serializable
    class Example:
        arg: int = 0

        def __post_init__(self):
            pass

3. Sonosco will generate ``__serialize__`` method for you, however this should not be used directly.
   Instead use ``sonosco.serialization.Serializer.serialize`` method:

::

    Serializer().serialize(Example(), "/path/to/save/object")

***************
Deserialization
***************

In order to deserialize class use: ``sonosco.serialization.Deserializer.deserialize`` method.

::

    ex = Deserializer.deserialize(Example, "/path/to/save/object")

Only object serialized with sonosco ``@serializable`` can be deserialized!

That's it. To see a full example, check out this `serialization example <https://github.com/Roboy/sonosco/blob/master/tests/test_serialization.py>`_ or
check one of the `models <https://github.com/Roboy/sonosco/tree/master/sonosco/models>`_ in the sonosco repository.

Check also inline docs for more details about different features  of (de)serialization.