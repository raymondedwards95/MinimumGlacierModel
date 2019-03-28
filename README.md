# MinimumGlacierModel

Code for a Minimum Glacier Model (MGM). Here it is applied to the glacier Veteranen on Spitsbergen.

## Usage
First import the modules:
```
from glacier import *
```
Then a glacier can be 'created' with
```
glacier = LinearBedModel(b0, s)
glacier = ConcaveBedModel(b0, ba, xl)
glacier = CustomBedModel(x, y)
```
where the first line creates a model for a linear bed with maximum bed elevation `b0` and slope `s`, 
the second line creates a model for an exponentially decreasing bed with top elevation `b0`, final elevation `ba` and length-scale `xl`
and the third line creates a model for a custom bed given by two arrays `x` with x-coordinates and `y` with corresponding y-coordinates.
Other parameters are for example the ice parameters `alpha`, `beta` and `nu` and the calving parameters `c` and `kappa`.

The glacier can be simulated in time (using a Forward Euler scheme) by using
```
glacier.integrate(dt, time, E)
```
where `dt` is the timestep, `time` is the simulated time and `E` is the equilibrium line altitude.
The result can be plotted using the data in `glacier.t` for time and `glacier.L` for glacier length or using the function
```
glacier.plot()
```

## Application to Veteranen
_For full explanation, see the notebooks._

![Bed of Veteranen](/figures/bed.png)
![History of Veteranen](/figures/veteranen_past_4000.png)

