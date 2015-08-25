This is the code used to produce the results published in

Betz T, Shapley R, Wichmann F A, Maertens M (2015) Testing the role of
luminance edges in White's illusion with contour adaptation. Journal of Vision.

The repository contains:
 * the code for running the experiment (experiment.py); requires hrl to run.
 * code for the BIWaM model (CIWaM)
 * code for the FLODOG model (f_l_odog_models)
 * code to test contour adaptation in BIWaM (analyze_biwam_adaptation.m) 
 * code to test contour adaptation in FLODOG (evaluate_adaptation.m) 
 * code to test contour adaptation in ODOG, and analyze adaptation results for all models (model_adaptation.py); requires lightness_models to run
 * a script to analyze the experimental data (analyze_contour_data.py);
   requires `ocupy <https://github.com/nwilming/ocupy>`_
 * the experimental data (exp_data)
 * a script to generate contour adaptation demo videos


Get in touch with `Torsten
<http://www.cognition.tu-berlin.de/menue/tubvision/people/torsten_betz/>`_
in case you have questions.
I apologize for all the imperfections in this code that will make it difficult
to run it as it is (such as absolute paths, missing input files, poor
documentation). I realized that if I try to fix everything before I publish it,
I might never publish the code at all, and I believe the current version is still better than
nothing for anyone interested in reproducing our results.
