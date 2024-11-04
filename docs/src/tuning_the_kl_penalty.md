# Tuning the KL penalty

## First sweep

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

The curves are in roughly the right order: A higher `beta` 
shrinks the raw reward.

The KL penalty (before multiplying by `beta`):

<figure>
  <img src=assets/sweep_kl_penalty.png alt=""/>
  <figcaption>The x-axis is number of steps</figcaption>
</figure>

Again, this makes sense: When the penalty coefficient `beta` is small,
gradient descent doesn't optimize for the KL penalty.

last generation for each

Let's look at the last two text generations[^note] for each train:

- `beta=0`
  - Last: "Once upon a time to the tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall tall"
  - Second-to-last: The same, except missing "the"
- `beta=0.01`
    - Last: "Once upon a time to to the tree to the tree to the tree to the tree to the tree to the tree to the tree to the tree to the tree to the tree to the tree to the tree to the tree to to the tree to the tree to the tree to the tree to the tree to the tree to the tree to the tree to to the tree to the tree to the tree to the tree to the tree to the tree to the tree to to the tree to the tree to the"
    - Second-to-last: The same, except with one more "to"
- `beta=0.1` 
    - Last: "Once upon a time there saw something she saw she suddenly suddenly she saw she she suddenly saw suddenly she she she saw she suddenly she she suddenly she she suddenly she saw she she suddenly she saw suddenly she she suddenly suddenly she saw suddenly she suddenly saw she suddenly she saw suddenly she she suddenly she suddenly she saw suddenly she suddenly she she suddenly saw she suddenly she suddenly she she she saw suddenly she she suddenly she suddenly suddenly she saw suddenly she suddenly she she saw suddenly she suddenly saw she suddenly she saw suddenly"
    - Second-to-last: The same, except with one more "she"
- `beta=1` 
    - Last: "Once upon a time there was something special something special something special something special something special something special something special something special. Suddenly she saw something special something special something special something special something special. Suddenly she saw something special something special something special something special she saw something special something special something special. Suddenly she saw something special something special something special something special something special something special she saw something special. Suddenly she saw something special something special something special something special she saw something special something special she saw special something special."
    - Second-to-last: The same, except the sentences aren't exactly the same lengths
- `beta=10` 
    - Last: "Once upon a time there was a girl. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something. She saw something"
    - Second-to-last: The same
- `beta=100`
    - Last: "Once upon a time, there was a little girl. She was very independent and loved to help her mom. One day, she noticed that there was something scary backstage in the library. The little girl was frightened by the display at the building, so she asked her mom to stay close and make sure she was safe. Her mother was teaching her how to stay calm and how to make the noise strongly, so that the images in the theater were not scary."
    - Second-to-last: "Once upon a time, there was a little girl who liked to play in the rain. Her parents told her that it wasn't safe, but the little girl couldn't help but be curious. One day, the girl's parents said they had a surprise for her. When they revealed it was a brand new toy. The girl couldn't believe her ears. She quickly hugged her parents and thanked them for the surprise. From then on, the little girl never stopped playing in the"
    
Conclusions: As `beta` increases, the text generations 
become more grammatically correct. From `beta=10` to `beta=100`, 
the generated text has a step change: The model is no
longer writing the same few words over and over. But also at `beta=100`, 
the model isn't very good at alliteration.

You can see this in the reward curve as well. The violet curve
(`beta=100`) has a much lower raw reward, and also its reward varies from 
step to step, since it's not always generating the same text.

We'd like varied text and alliteration. 
Hence the optimal value of `beta` is somewhere between 10 and 100.


## Second sweep



[^note]: Technically these are the last two text generations displayed
by TensorBoard, which hides some data points because of its reservoir sampling.

