# Tuning the KL penalty

We still don't know what the KL penalty coefficient `beta`
should be. Let's sweep over
the following values:

- `beta=0` (red)
- `beta=0.01` (orange)
- `beta=0.1` (yellow)
- `beta=1` (green)
- `beta=10` (blue)
- `beta=100` (violet)


<!--  

20241030-225813soft-crafty-quetzal-of-tolerance


20241031-000513tall-opal-silkworm-from-hell


20241031-011205greedy-scrupulous-labradoodle-of-awe


20241031-021901rough-hypnotic-bug-of-ampleness


20241031-032527quizzical-muscular-cockatoo-from-ganymede


20241031-043206optimistic-deft-starfish-of-radiance

-->


The raw reward:

<figure>
  <img src=assets/sweep_raw_reward.png alt=""/>
  <figcaption>The x-axis is number of steps</figcaption>
</figure>

The KL penalty (before multiplying by `beta`)

<figure>
  <img src=assets/sweep_kl_penalty.png alt=""/>
  <figcaption>The x-axis is number of steps</figcaption>
</figure>

Again, this makes sense: When the penalty coefficient `beta` is small,
gradient descent doesn't optimize for the KL penalty.

last generation for each

optimal value of `beta`
is
somewhere between
