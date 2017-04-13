==================================
Install TensorFlow from the source
==================================

.. _TensorFlow: https://www.tensorflow.org/install/
.. _Installing TensorFlow from Sources: https://www.tensorflow.org/install/install_sources

The installation is available at `TensorFlow`_. Installation from the source is recommended becasue the user can build the desired TensorFlow binary for the specific architecture. It enrich the TensoFlow with a better systam compatibility and it will run much faster. Instaling from the source is available at `Installing TensorFlow from Sources`_ link. The official TensorFlow explanations are concise and to the point, however few things might become important as we go through the installation. We try to project the step by step to avoid any confusion. The following sections must be considered in the written order.

The assumption is that installing TensorFlow in the ``Ubuntu`` using ``GPU support`` is desired.

------------------------
Prepare the environment
------------------------

The following should be done in order:
    
    * Bazel installation
    * TensorFlow Python dependencies installation
    * TensorFlow fGPU prerequisites setup

~~~~~~~~~~~~~~~~~~~
Bazel installation
~~~~~~~~~~~~~~~~~~~
