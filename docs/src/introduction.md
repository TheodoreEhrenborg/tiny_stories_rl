# Introduction

This project uses reinforcement learning to encourage
the language model `roneneldan/TinyStories-33M` 
to generate stories with alliteration.

Code [here](https://github.com/TheodoreEhrenborg/tiny_stories_rl)

## Demo

Here's a non-cherry-picked generation from the unmodified model:

<!-- step 4: This is the first one from gorgeous-cream-jackal -->

> Once upon a time there was a cute little bunny named Fluff. Fluff was a very clever bunny who liked to explore and find new places every day. One day while Fluff was hopping around, he noticed a big knot in a tree. He wanted to attach the knot to something, so he used his paws to attach it. Fluff hopped up to the nearest rabbit and said, "Hello, can you help me attach this knot to the tree?" The

By coincidence, there are some consecutive words with the same initial letter,
such as 
`Fluff. Fluff`
and
`to the tree?" The`.

Here's a generation from the model after reinforcement learning:

<!-- step 5,079 --- this is the first one from busy-flawless-markhor after step 5000 -->

> Once upon a time, there was a tall tree. The tall tree was in the middle of a park. One day, a little girl wanted to touch the tall tree. She started to walk towards the tall tree. When she got to the tall tree, she started to touch the branches. Suddenly, the tree started to grow taller and taller! The little girl was amazed and she ran away from the tall tree. She kept running and she forgot about the tall tree. She

(Here I chose the first generation after step 5000, 
so this is a little cherry-picked.)

Now there's much more alliteration, such as 
`to touch the tall tree`,
`She started`,
and
`towards the tall tree`.

```admonish
For simplicity, I'm defining alliteration 
as two or more consecutive words with the same
initial letter. So "Zebadiah the Zebra" doesn't count
because of the intermediate "the", and
"seven cycling psychologists" doesn't count 
even though the initial sound is the same.
```
