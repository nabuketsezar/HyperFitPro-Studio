# HyperFitPro Studio Embedded Plugin Pack

This release adds an embedded plugin pack under:

```text
hyperfitpro/plugins/embedded_models/
```

The models in this folder are loaded through the same plugin mechanism as user plugins, but they are shipped inside the application package. They appear in the model selector with model numbers `9010` through `9022`.

## Embedded plugin models

| No. | Plugin model | Main use |
|---:|---|---|
| 9010 | One-term Ogden | Compact power-law baseline beyond Neo-Hookean |
| 9011 | Demiray exponential | Exponential stiffening / soft-tissue type response |
| 9012 | Carroll high-strain | High-strain invariant stiffening with I2 contribution |
| 9013 | Lopez-Pamies two-term | I1 power-law two-term calibration |
| 9014 | Isihara three-parameter | Compact polynomial I1/I2 extension |
| 9015 | Haines-Wilson five-parameter | Multi-mode polynomial fitting |
| 9016 | Biderman four-parameter | Yeoh-like cubic I1 plus I2 term |
| 9017 | Gent-Thomas-Yeoh hybrid | I1 polynomial plus logarithmic I2 sensitivity |
| 9018 | Exponential-polynomial I1/I2 | Combined exponential stiffening and polynomial response |
| 9019 | Shifted power-law mixed | Shifted invariant power-law with I2 correction |
| 9020 | Reduced polynomial 4-term | Yeoh-style four-term I1-only plugin |
| 9021 | Gent-Yeoh finite extensibility | Limiting-chain + Yeoh correction hybrid |
| 9022 | Treloar I1 series | Compact I1 series plugin |

## Loading behavior

The registry now loads models in this order:

1. Built-in table models `1` through `42`.
2. Embedded plugins from `hyperfitpro/plugins/embedded_models`.
3. User plugins from configured plugin folders and `HYPERFITPRO_PLUGIN_PATH`.

If two models use the same model number, the first one loaded is kept. Therefore built-in table models are protected from plugin number collisions.

## Creating your own plugin

Use:

```bat
python -m hyperfitpro.cli --export-plugin-template "%USERPROFILE%\.hyperfitpro_studio\plugins\my_model.py"
```

Then edit the generated `MODEL_CLASS`. The only mandatory pieces are:

- subclass `HyperelasticModel`,
- set a unique `number`,
- define `parameter_specs`,
- define `equation_latex`,
- implement `W(self, stretches, p)`,
- expose `MODEL_CLASS = YourClassName`.

## Verification

Run:

```bat
python verify_embedded_plugins.py
```

Expected result:

```text
Embedded plugin verification passed.
```
