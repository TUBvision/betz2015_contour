addpath CIWaM/


alphas = [.5, .2, 0];
mus = linspace(.1, .0005, 200);
sigmas = linspace(.005, .0001, 8);
results = nan(length(alphas), length(mus), length(sigmas));

% determine test patch location
stimulus = imread('../model_input/dec.png');
padded = zeros(512);
padded(143:370, 143:370) = stimulus;
%patch_idx = padded == 127;
patch_idx = sub2ind([512, 512], [275, 276, 275, 276], [256, 256, 257, 257]);


for stim = {'dec', 'inc'}            
    % load stimulus
    stimulus = double(imread(sprintf('../model_input/%s.png', stim{1})));
    % undo int encoding
    stimulus(stimulus == 127) = .5;
    stimulus(stimulus == 114) = .45;
    stimulus(stimulus == 140) = .55;
    stim_padded = ones(512) * .5;
    stim_padded(143:370, 143:370) = stimulus;
    for adaptor = {'ortho', 'para', 'ortho_shifted', 'para_shifted'}%,'none'
        display(adaptor{1})

        if ~strcmp(adaptor{1}, 'none')
            adapt_stim = double(imread(sprintf('../model_input/%s.png', adaptor{1})));
            adapt_stim(adapt_stim == 127) = .5;
            adapt_padded = ones(512) * .5;
            adapt_padded(143:370, 143:370) = adapt_stim;
            
        end
        for a = 1:length(alphas)
            alpha = alphas(a);
            for b = 1:length(mus)
                mu = mus(b);
                for c = 1:length(sigmas)
                    sigma = sigmas(c);
                    if ~strcmp(adaptor{1}, 'none')
                        ind=CIWaM(stim_padded, [5,4], 7, 1, 0, 3.36, adapt_padded, alpha, mu, sigma);
                    else
                        ind=CIWaM(stim_padded, [5,4], 7, 1, 0, 3.36);
                    end
                    patch_lightness = mean(ind(patch_idx));
                    results(a, b, c) = patch_lightness;        
                    
                end
            end
        end
    save(sprintf('../data/biwam/%s_%s.mat', stim{1}, adaptor{1}), 'results', 'alphas', 'mus', 'sigmas')
    end
end
