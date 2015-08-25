addpath f_l_odog_models/
for stim = {'dec', 'inc'}
    for adaptor = {'none', 'ortho', 'para', 'ortho_shifted', 'para_shifted'}
        fprintf(adaptor{1})
        analyze_adaptation(stim{1}, adaptor{1}, [.5, .2, 0], [.08, .01, .001], [.04, .005, .0001], true);
    end
end