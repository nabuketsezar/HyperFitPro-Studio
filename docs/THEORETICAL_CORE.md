# HyperFitPro theoretical / mechanical core update

This release replaces the previous path-derivative stress approximation with a
mechanically consistent incompressible isotropic response kernel.

## Implemented constitutive response

For every active hyperelastic model, the model file still defines only the strain
energy density

\[
W = W(\lambda_1,\lambda_2,\lambda_3)
\]

and the shared kernel computes principal derivatives and stresses as

\[
\sigma_i = \lambda_i \frac{\partial W}{\partial \lambda_i} - p
\]

\[
P_i = \frac{\partial W}{\partial \lambda_i} - \frac{p}{\lambda_i}
\]

where \(p\) is the incompressibility pressure determined from the traction-free
principal direction.

## Test-mode equations used internally

### Uniaxial tension/compression

\[
\lambda_1 = \lambda, \qquad \lambda_2=\lambda_3=\lambda^{-1/2}
\]

\[
p = \lambda_2 \frac{\partial W}{\partial \lambda_2}
\]

\[
P_{11}=\frac{\partial W}{\partial \lambda_1}-\frac{p}{\lambda_1}
\]

### Equibiaxial tension

\[
\lambda_1=\lambda_2=\lambda, \qquad \lambda_3=\lambda^{-2}
\]

\[
p = \lambda_3 \frac{\partial W}{\partial \lambda_3}
\]

\[
P_{11}=P_{22}=\frac{\partial W}{\partial \lambda_1}-\frac{p}{\lambda_1}
\]

### Planar tension / pure shear

\[
\lambda_1=\lambda, \qquad \lambda_2=1, \qquad \lambda_3=\lambda^{-1}
\]

\[
p = \lambda_3 \frac{\partial W}{\partial \lambda_3}
\]

\[
P_{11}=\frac{\partial W}{\partial \lambda_1}-\frac{p}{\lambda_1}
\]

### Simple shear

The principal stretches of simple shear are

\[
\lambda_{1,2}=\sqrt{1+\frac{\gamma^2}{4}} \pm \frac{\gamma}{2}, \qquad \lambda_3=1
\]

The shear response is obtained by transforming the principal Cauchy stresses
back to the laboratory axes:

\[
\sigma_{12}=\frac{\lambda_1 W_{,1}-\lambda_2 W_{,2}}{\sqrt{\gamma^2+4}}
\]

## Compressibility / volumetric extension

The source table is incompressible. adds an optional volumetric fitting branch
whenever a volumetric dataset is imported. The default volumetric penalty is

\[
U(J)=\frac{1}{2}K(J-1)^2
\]

with hydrostatic response

\[
p_J = K(J-1)
\]

The fitted output includes \(K\). Mapping \(K\) to FEA-specific \(D_i\) or bulk
penalty coefficients must be checked against the chosen solver convention.

## Stability / admissibility scan

adds a shared stability module checking:

- finite energy and stress response,
- near-zero reference energy,
- positive small-strain shear modulus,
- non-negative energy relative to reference over deformation paths,
- tangent stiffness positivity over uniaxial, biaxial, planar and simple-shear scans.

This is a strict engineering screen, not a formal symbolic strong-ellipticity
proof for every possible black-box strain-energy expression.
