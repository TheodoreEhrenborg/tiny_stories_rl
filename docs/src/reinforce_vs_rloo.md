# REINFORCE vs RLOO

```note
For this chapter only, I'll be using a simpler
reward function: The reward is the number of words
in the generation that start with "A" or "a".
```

## How is reinforcement learning different from supervised learning?

In supervised learning, we want to train the model 
to accurately predict labels given inputs.
For instance, the input might be "the position of all the pieces
on a chess board", and the label might be 
"which side has the advantage, as estimated by a human judge".
The dataset looks like a list of `(input, label)` tuples.

In reinforcement learning, we instead want to train the
model to produce good actions, as measured by some reward
function. The action might be "go play a chess game",
and the reward would be "did you win?"
Typically there isn't a fixed dataset of `(action, reward)` tuples, 
since the model is taking the actions on the fly.

In my case, the action is "write a story with 100 tokens", and the
reward is "how many words started with A or a?"

## The REINFORCE algorithm
If the model produces a story with a high reward, we want
to nudge the model to generate stories more like that one.


A simple way is to feed the generated text to the model
to get the model's probability of seeing the text in the wild.
Then we use backpropagation to get the gradient of that
probability with respect to the model weights.
If we add `gradient` to the weights, we'll
the model more likely to generate this text sequence,
and in theory also more likely to generate similar text sequences.
We want to nudge the model more if the reward was higher (and vice versa),
so instead we add `reward * gradient` to the weights.
This is the REINFORCE algorithm.

(The [REINFORCE paper](https://link.springer.com/article/10.1007/BF00992696) 
has variants on this algorithm, so I'm simplifying.)

```note
For convenience we use the cross-entropy loss to 
represent the model's probability of seeing a text. 
Hence we have to be careful with the sign---if we want to 
increase the probability of a text sequence, 
we should add the gradient with the right sign so that 
the cross-entropy loss decreases.
to increase

```

```note
PyTorch doesn't have the best support for 
directly adding gradients to a model's weights 
(although it can be done with some hacks).
It's easier to instead use `torch.optim.SGD` as a middleman,
which has the same effect of adding a gradient to all model weights. 
```



## The RLOO algorithm

"play a video game"
"the final score"


"generate text autoregressively"
