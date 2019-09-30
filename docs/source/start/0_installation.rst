.. _installation:
Installation
===============

Via pip
^^^^^^^^^

The easiest way to use Sonosco's functionality is via pip:
::
    pip install sonosco

**Note**: Sonosco requires Python 3.6 or higher.

For reliability, we recommend using an environment virtualization tool, like virtualenv or conda.


For developers
^^^^^^^^^^^^^^

Clone the repository and install dependencies:
::
    # Clone the repo and cd inside it
    git clone https://github.com/Roboy/sonosco.git && cd sonosco

    # Create a virtual python environment to not pollute the global setup
    python -m venv venv

    # Activate the virtual environment
    source venv/bin/activate

    # Install normal requirements
    pip install -r requirements.txt

    # Link your local sonosco clone into your virtual environment
    pip install -e .

Now you can check out some of the :ref:`Getting Started <start_training>` tutorials, to train a model or use
the transcription server.