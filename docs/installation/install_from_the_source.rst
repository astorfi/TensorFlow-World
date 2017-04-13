==================================
Install TensorFlow from the source
==================================

.. _TensorFlow: https://www.tensorflow.org/install/
.. _Installing TensorFlow from Sources: https://www.tensorflow.org/install/install_sources
.. _Bazel Installation: https://bazel.build/versions/master/docs/install-ubuntu.html
.. _CUDA Installation: https://github.com/astorfi/CUDA-Installation
.. _NIDIA documentation: https://github.com/astorfi/CUDA-Installation



The installation is available at `TensorFlow`_. Installation from the source is recommended becasue the user can build the desired TensorFlow binary for the specific architecture. It enrich the TensoFlow with a better systam compatibility and it will run much faster. Instaling from the source is available at `Installing TensorFlow from Sources`_ link. The official TensorFlow explanations are concise and to the point, however few things might become important as we go through the installation. We try to project the step by step to avoid any confusion. The following sections must be considered in the written order.

The assumption is that installing TensorFlow in the ``Ubuntu`` using ``GPU support`` is desired. ``Python2.7`` is chosen for installation.

------------------------
Prepare the environment
------------------------

The following should be done in order:
    
    * Bazel installation
    * TensorFlow Python dependencies installation
    * TensorFlow GPU prerequisites setup

~~~~~~~~~~~~~~~~~~~
Bazel installation
~~~~~~~~~~~~~~~~~~~

Please refer to `Bazel Installation`_.

``WARNING:`` The Bazel installation may change the supported kernel by the GPU! After that you may need to refresh your GPU installation or update it other wise you may get the following error when evaluating the TensorFlow installation:

.. code:: bash

    kernel version X does not match DSO version Y -- cannot find working devices in this configuration
    
For solving that error you may need to purge all NVIDIA drivers and install or update them again. Please refer to `CUDA Installation`_ for further detail.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TensorFlow Python dependencies installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For installaion of the required dependencies, the following command must be executed in the terminal:

.. code:: bash

    sudo apt-get install python-numpy python-dev python-pip python-wheel
    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TensorFlow GPU prerequisites setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following requirements must be satisfied:

    * NVIDIA's Cuda Toolkit and its associated drivers(version 8.0 is recommended). The installation is explained at `CUDA Installation`_.
    * The cuDNN library(version 5.1 is recommended). Please refer to `NIDIA documentation`_ for further details.
    * Installing the ``libcupti-dev`` using the following command:
    
    .. code:: bash

    sudo apt-get install libcupti-dev
    
 




