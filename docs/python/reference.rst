Python API Reference
====================

The ``uform`` package is one compiled extension, ``uform.lib``, re-exported whole.
``uform/lib.pyi`` carries the typed signatures of everything the extension defines.

======
Models
======

.. autoclass:: uform.Model
    :members:
    :undoc-members:

.. autoclass:: uform.ModelDescription
    :members:

.. autoclass:: uform.Encoded
    :members:

.. autoclass:: uform.TextEncodeStats
    :members:

.. autoclass:: uform.ImageEncodeStats
    :members:

.. autoclass:: uform.VideoEncodeStats
    :members:

.. autoclass:: uform.AudioEncodeStats
    :members:

==================
Executors & Arenas
==================

.. autoclass:: uform.Executor
    :members:
    :undoc-members:

.. autoclass:: uform.Arena
    :members:
    :undoc-members:

.. autoclass:: uform.ArenaBuffer
    :members:

.. autoclass:: uform.RowsTape
    :members:

.. autoclass:: uform.On
    :members:

.. autoclass:: uform.DeferredOn
    :members:

.. autoclass:: uform.Deferred
    :members:

========
Training
========

.. autoclass:: uform.Mesh
    :members:
    :undoc-members:

.. autoclass:: uform.MeshOptions
    :members:

.. autoclass:: uform.MeshDescription
    :members:

.. autoclass:: uform.Trainer
    :members:
    :undoc-members:

.. autoclass:: uform.TrainerStepStats
    :members:

.. autoclass:: uform.BatchShape
    :members:

.. autoclass:: uform.ImageShape
    :members:

.. autoclass:: uform.VideoShape
    :members:

.. autoclass:: uform.AudioShape
    :members:

.. autoclass:: uform.Masking
    :members:

.. autoclass:: uform.Contrastive
    :members:

.. autoclass:: uform.Schedule
    :members:

.. autoclass:: uform.Relations
    :members:

.. autoclass:: uform.Edges
    :members:

==========
Vocabulary
==========

.. autofunction:: uform.version

.. autofunction:: uform.capabilities_compiled

.. autofunction:: uform.capabilities_available

.. autoclass:: uform.Status
    :members:
    :undoc-members:

.. autoclass:: uform.Scalar
    :members:
    :undoc-members:

.. autoclass:: uform.Mood
    :members:
    :undoc-members:

.. autoclass:: uform.Outputs
    :members:
    :undoc-members:

.. autoclass:: uform.Modality
    :members:
    :undoc-members:

.. autoclass:: uform.Objective
    :members:
    :undoc-members:

.. autoclass:: uform.Negatives
    :members:
    :undoc-members:

.. autoclass:: uform.MultimemTier
    :members:
    :undoc-members:

======
Errors
======

.. autoexception:: uform.Error
.. autoexception:: uform.OutOfMemoryError
.. autoexception:: uform.ShapeMismatchError
.. autoexception:: uform.UnsupportedLayoutError
.. autoexception:: uform.UnknownKindError
.. autoexception:: uform.InputOutputError
.. autoexception:: uform.DeviceError
.. autoexception:: uform.MalformedError
.. autoexception:: uform.InternalError
.. autoexception:: uform.ConfigMismatchError
.. autoexception:: uform.ContendedError
.. autoexception:: uform.CancelledError
.. autoexception:: uform.UnsupportedError
.. autoexception:: uform.InvalidArgumentError
