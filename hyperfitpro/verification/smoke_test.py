from hyperfitpro.core.registry import get_model, load_models
from hyperfitpro.core.tests import load_test_data
from hyperfitpro.core.optimizer import fit_model
from hyperfitpro.core.run_manager import create_run_folder, save_run

print('models', len(load_models(include_inactive=True)))
m=get_model(6)
d1=load_test_data('sample_data/neo_hookean_uniaxial.csv','uniaxial',1.0)
d2=load_test_data('sample_data/neo_hookean_biaxial.csv','biaxial',1.0)
res=fit_model(m,[d1,d2],method='least_squares_trf',max_evals=300,progress=print)
print(res)
folder=create_run_folder('hyperfit_test_smoke',m)
save_run(m,[d1,d2],res,folder)
print('saved', folder)
