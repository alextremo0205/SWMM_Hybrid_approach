# Memory helper

Unnormalize NN values when adding up to the node depth.

Side by side comparisons. Original, prediction, error.

Multiple rainfall events.

Runoff instead of rainfall

Performance metrics. Calculate the error of the time series as Elvin in Bishwadeep's presentation.
Statistics:    Number of parameters. Number of training examples.

Consider the invert offsets.

Consider two different networks depending on the pipe section.
(One for the EGG and other for the CIRCLE)

Check Saint-Vennant equations.

Heun or Runge-Kutta approach.

Metrics of performance.

Hyperoptimization of the Neural Network

Annotations:

## 2022-06-12 01:39:00

The time for running the simulations in SWMM (1 block rainfall, 10 synthetic cases, 10 real cases with intensity bigger than 10) was 5 minutes, 19 seconds

I removed the DWF information from the inp file.

There are two ways that the information can be used. One is for training the program (generating the sample objects)
The other is by passing an event (runoff) and an initial depth to see how it solves it.

## 2022-06-12 09:59:42

I improved the database code: Moved the extract heads and runoff method to the utils part.

## 2022-06-13 07:03:51

Remember to include the nodal area (also for the future storage tanks)

## 2022-06-13 08:44:31

Debe simular la profundidad en todos los nodos. Especialmente en los outfalls. Cómo sé cuánto caudal pasa?
La condición de frontera es cero, así que debe ser función del valor que se está rechazando, el que se está volviendo cero.

## 2022-06-13 15:19:31

Hay un problema con las diferencias de nivel, su magnitud es mayor a uno, no es bueno para entrenar las redes neuronales.

## 2022-06-14 10:10:23

Pass the diameter multiple times to the ANN that calculates dh due to gradient.

## 2022-06-15 10:10:59

Include the manhole area as node attribute.

## 2022-06-17 12:23:41

For some reason, it is returning a number 2 in the mask values.

## 2022-06-20 09:57:43

Update: The reason for this was that the 2 meant that 2 pipes were going to flow to a single node.
                        The number 2 appeared as the sum of the neighbors. It was on the node space, not the pipe space.

## 2022-06-20 09:59:03

Validation!!
Program the unrolling
How good is the model working?
Visualize in equal terms
Control + Z to the parameters

## 2022-06-23 10:44:52

Validation is implemented. After this, do not touch the database. Better, only add examples.
Before going to visualization, I'll implement the new version of is_giver_manhole_dry
Expanding features
Priority: Visualization of unnormalized results.

## 2022-06-24 10:22:01

I need the rolled out version of the depths
Include t-1 as Roberto recommended
Sliding window instead of separated windows
Reduce the complexity of the auxiliary MLPs

## 2022-06-24 10:38:04

Apparently there is an error with the window creation. It is not sliding correctly.

## 2022-06-27 08:51:54

    Fixed, I reduced the maximum length so that it trims the sliding before it gets to the end and creates a smaller size window.

## 2022-06-27 08:52:47

I could only give windows with changes
I could make the ANN to be uneven and symmetric.

## 2022-06-27 19:56:33

Check for the flow conditionality
Try different approaches for the link interchange
    Make it symmetric
Add more than one previous time step

## 2022-06-29 09:53:36

Pickle the training windows for faster access

## 2022-06-30 09:48:53

Add roughness? Geom_2?

Based on conversation with Roberto:
    - Put multiple steps behind -> Priority
    - Early stopping is a must (Implemented)
    - Saving the model with the best validation error (Coded)
    - Validation should be on the entire events (Implemented)
    - Possible weighted loss for time steps ahead

Try concatenating instead of adding

Add alpha*inputs
Add matrices that multiply the sum of message passing and the MLP of (h and runoff) (This option is the preference by Roberto)
Or add MLP of concatenated (ht, inputs)

Based on conversation with Job and João.
I could post-process the heads so that they are above ground. Not in the training so that the loss function does not pick up on that.
I could also include DWF so that the network sees constant inflow.

Based on conversation with Riccardo:
Check residual connections as done by Xing and Sela
Double check the temporality
Check if the model works without considering the xi and xj at the pipes. Or maybe with the difference of them.
Is there a way to include the flows? Predict them? Or in the loss function?

## 2022-07-01 10:07:01

Distinguish between outfalls and junctions with one hot vector
Presenter class
Session class

Visualize the heads with the inverted hietograph
