
# Literature and engineering basis used in HyperFitPro

The program uses the equations digitized from the provided table. The bounds are implemented as practical optimization search ranges. They are intentionally broad and scaled by the uploaded data stress level where appropriate.

## Test types

Recommended calibration data for isotropic hyperelasticity generally includes:

- uniaxial tension/compression;
- equibiaxial tension/compression;
- planar tension / pure shear;
- volumetric test data if compressibility must be calibrated.

Abaqus documentation uses these same categories for hyperelastic material calibration. It also emphasizes that uniaxial, biaxial, and planar data should be ordered by nominal strain and that model behavior should be assessed by comparing predictions with experimental data.

## Bounds policy

There is no universal upper/lower bound set that applies to all elastomers and soft tissues. For this reason the software uses:

- stress-like coefficients scaled by the measured stress scale;
- positive lower limits for shear-modulus-like parameters;
- domain-enforcing bounds for limiting-chain parameters such as `I_L` and `J_L`;
- dimensionless bounds for exponential, power-law, and Ogden exponent parameters.

The bounds are visible in the GUI and can be edited in each model file.
