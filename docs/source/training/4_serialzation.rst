.. _serialisation:
Serialisation Guide
=====================

In order to enable the serialization of a model, the following steps need to be followed.
1. add the ``@serializable`` decorator on top of the class.
This will automatically add a ``__init__()`` function, so you...
2. need to specify all required parameters that need to be initialized write after the model description
**with type annotations!!**

That's it. To see a full example, check out this `serialization example <https://github.com/Roboy/sonosco/blob/master/tests/test_serialization.py>`_ or
check one of the `models <https://github.com/Roboy/sonosco/tree/master/sonosco/models>`_ in the sonosco repository.
