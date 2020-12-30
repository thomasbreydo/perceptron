perceptron-rs
=============

This is a mini-project in which I create a perceptron from scratch in Rust and create bindings to use it in Python. The
perceptron learns using the perceptron learning rule.

What is a perceptron?
---------------------

In very simple terms, a perceptron is a binary classifier that takes a vector
input and outputs a 0 or a 1.

How does a perceptron work?
---------------------------

Parts of a perceptron
~~~~~~~~~~~~~~~~~~~~~

A perceptron has 2 tunable parameters:

- A vector of **weights**

- A **bias**

Classifying an input
~~~~~~~~~~~~~~~~~~~~

Given an input (assuming the input shape matches the shape of the perceptron's
weights), classification is performed as follows:

1. Calculate the dot product of the perceptron's weights and the input vector.

2. Add the perceptron's bias to the dot product.

Finally, classify the point as 1 if the total sum is positive. Otherwise,
classify it as a 0.

Initialization
~~~~~~~~~~~~~~

Before a perceptron is "trained" to classify your data well, it must first be
given some set of weights and some bias. During the
`learning process`_ (below), these weights and this bias will
be altered.

Given enough time and a **linearly separable** dataset, a perceptron will
always converge, no matter what its starting weights were. In this program,
we'll initialize :code:`weights` to :code:`[1, 1, ..., 1]` and :code:`bias` to :code:`0`.

Learning process
~~~~~~~~~~~~~~~~

Learning consists of repeating an identical process a total of :code:`n` times where
:code:`n` is the number of *epochs*. Learning also requires a *learning rate*, :code:`learning_rate`,
which specifies how sloppily-but-quickly or carefully-but-slowly our perceptron
arrives at an ideal :code:`weights` and :code:`bias`.

At each *epoch*, the following adjustments are made for each sample (simplified):

1. If the sample is currently being classified correctly, change no weights.

2. Otherwise,

    2.1. If the sample is of type :code:`0` but is being classified :code:`1`, increase
    the weights.

    2.2. If the sample is of type :code:`1` but is being classified :code:`0`, decrease
    the weights.

