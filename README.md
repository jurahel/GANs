# GANs
The concept of GANs was firstly introduced by Ian Goodfellow in the context of creating
artificial pictures from an underlying population by approximating the distribution of their
pixels.
In the concept of GANs, a generative neural network (generator) proposes a random
candidate, here a distribution. Subsequently, a discriminative network (discriminator) judges
the likelihood of the candidate belonging to the population. In case a candidate resembles the
population, it achieves a high score in terms of a probability.
![GAN-Scheme](https://github.com/jurahel/GANs/blob/master/GAN_scheme.png)

Thus, the goal of the generator is to create samples, which the discriminator cannot distinguish from the population. In the
course of training, both the discriminator and the generator improve and finally latter samples.
data which is close to the underlying population but ideally different with respect to the outcomes.
In the initial application of Goodfellow et al. the pixel distribution formed artificial
faces, which shows the ability of GANs to sample from complex, highly conditional and patterned
distributions. This proposes their usage in applications with similar challenges.
