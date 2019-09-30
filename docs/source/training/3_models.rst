.. _models:

Models
========
Sonosco provides some predefined deep speech recognition models, that can be used for training:

Deep Speech 2
^^^^^^^^^^^^^^

We took the pytorch implementation of the `Deep Speech 2 <https://arxiv.org/abs/1512.02595>`_ model from `Sean Naren <https://github.com/SeanNaren/deepspeech.pytorch>`_
and ported it to the sonosco serialization guidelines.

Listen Attend Spell (LAS)
^^^^^^^^^^^^^^^^^^^^^^^^^^

We took the pytorch implementation of the `Listen Attend Spell <https://arxiv.org/abs/1508.01211v2>`_ model from `AzizCode92 <https://github.com/AzizCode92/Listen-Attend-and-Spell-Pytorch>`_
and ported it to the sonosco serialization guidelines.
This model can be imported using: ``from sonosco.models import Seq2Seq``

Sequence-to-Sequence Model with Time-Depth Separable Convolutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implemented a sequence-to-sequence model with Time-Depth separable convolutions in pytorch, following `a paper from Facebook AI <https://arxiv.org/abs/1904.02619>`_ .


These models can be simply imported and used for training. (See :ref:`Train your first model <start_training>` )
