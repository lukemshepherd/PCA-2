# Functions

## `mag(v)`

Finds magnitude of vector

## `angle (vector1, vector2)`

Finds the angel between two vectors

## `quaternion_rotation_from_angle(v, c_axis, theta)`

`v` = vector (np.array 1x3)

`c_axis` = cross product between two principle conponets 

`theta` = angle of rotation (radians) 

## `rotate(bone_f1, bone_f2, interpolate=False, scale_factor=2)`

Alines and rotates bone_f1 to bone_f2

`bone` = bone class object

`interpolate` = (boolean) if `True` bone_f1 is upscaled, rotated and downscaled to increase point density.

`scale_factor` = set how much the bone will be upscaled to increase the density.