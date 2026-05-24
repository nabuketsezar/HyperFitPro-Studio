from hyperfitpro.core.tests import load_test_data, auto_balance_dataset_weights

# Two-column processed stress-strain import
ss = load_test_data('sample_data/neo_hookean_uniaxial.csv', 'uniaxial', 1.0)
assert len(ss.x) >= 2
assert ss.diagnostics['import_kind'] in ('stress_strain', 'auto')

# Force-displacement conversion: area=10 mm2, gauge=20 mm
fd = load_test_data(
    'sample_data/neo_hookean_force_displacement.csv',
    'uniaxial',
    1.0,
    data_kind='force_displacement',
    area=10,
    gauge_length=20,
    force_unit='N',
    length_unit='mm',
)
assert len(fd.x) >= 10
assert abs(fd.x[-1] - 0.5) < 1e-10
assert fd.stress[-1] > 0

# Cyclic branch isolation
cyc = load_test_data('sample_data/neo_hookean_cyclic_uniaxial.csv', 'uniaxial', 1.0, branch='loading')
assert len(cyc.x) < 36
assert cyc.x[-1] >= 0.4

# Point weights and auto balancing
rw = load_test_data('sample_data/neo_hookean_uniaxial.csv', 'uniaxial', 1.0, region_weighting='balanced_bins')
assert rw.point_weight is not None
lst = auto_balance_dataset_weights([ss, fd])
assert all(d.weight > 0 for d in lst)
print('Data processing verification passed.')
print('stress-strain n=', len(ss.x), 'force-displacement n=', len(fd.x), 'cyclic-loading n=', len(cyc.x))
