# Open-Source MTG Card Sorting Robot
This is an attempt to build my own card sorting robot for MTG. All of the ones I have seen have impressed me but this is great effort to keep it closed source, which I do not believe benefits the MTG community. For years I have had this want to build a card sorting robot but it has been a slow and painful process. I am not working on the code for the sorting algo, and I have one working in my simulation of how the robot works. I will continue to improve the sorting algo and also will start to work on the other two important parts: the vision processing code, and the robot itself (code and physical robot)

The robot will operate like a 3d printer with a matrix of card stacks facing up. The cursor of the 3d printer will be fitted with a camera and vacuum so that it can pop and push amongst the stacks. this is just theory but I have seen similar robots made before.

Here is the sample output of the sorting algorithm given a full robot (1,475 cards):

```
Sorted 1475 items in 25 stacks with 59 elements per stack and a swap of 25 Stacks (50 total) in 62190 moves O(43n) (103327/79961  Uncached/Cached reads)
This requires at least a 7x8 grid in size and estimated 4.794872685185186 days to complete
```
