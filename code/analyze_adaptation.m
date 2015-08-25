function [output, adapted_responses] = analyze_adaptation(stim, adaptor, alphas, mus, sigmas, saveflag)
if nargin<6
    saveflag=false;
end
% compute model output with adaptation
stimulus = double(imread(sprintf('../model_input/%s.png', stim)));
% undo int encoding
stimulus(stimulus == 127) = .5;
stimulus(stimulus == 114) = .45;
stimulus(stimulus == 140) = .55;
[stim_y, stim_x] = size(stimulus);
if ~strcmp(adaptor, 'none')
    adapt_stim = double(imread(sprintf('../model_input/%s.png', adaptor)));
    adapt_stim(adapt_stim == 127) = .5;
end
%% prepare the model filters
PIXELS_PER_DEGREE = 31.277; %#ok<NASGU> The value is used inside BM_model_params
BM_model_params;
filters = cell(length(orientations), length(stdev_pixels));
for o = 1 : length(orientations)
    % loop over frequencies
    for f = 1 : length(stdev_pixels)
        % create the filter
        filters{o, f} = dog(model_y, model_x, stdev_pixels(f), ...
            stdev_pixels(f), 2, orientations(o) * pi/180);
    end
end

%% compute response to adapting stimulus
filter_response_adapt = nan(length(orientations), length(stdev_pixels), stim_y, stim_x);
if ~strcmp(adaptor, 'none')
    for o = 1 : length(orientations)
        % loop over frequencies
        for f = 1 : length(stdev_pixels)
            filter_response_adapt(o, f, :, :) = abs(ourconv(adapt_stim, filters{o, f}, 0.5));  % 0.5 pads image with gray
        end
    end
end

%% compute response to stimulus
filter_response_stim = nan(length(orientations), length(stdev_pixels), stim_y, stim_x);
for o = 1 : length(orientations)
    % loop over frequencies
    for f = 1 : length(stdev_pixels)
        filter_response_stim(o, f, :, :) = ourconv(stimulus, filters{o, f}, 0.5);  % 0.5 pads image with gray
    end
end

%%apply adaptation and FLODOG normalization
results = nan(length(alphas), length(mus), length(sigmas));
for a = 1:length(alphas)
    alpha = alphas(a);
    for b = 1:length(mus)
        mu = mus(b);
        for c = 1:length(sigmas)
            sigma = sigmas(c);
            if ~strcmp(adaptor, 'none')
                adapt_weights = (1 - (1 - alpha) .* normcdf(filter_response_adapt, mu, sigma));
                adapted_responses = filter_response_stim .* adapt_weights;
            else
                adapted_responses = filter_response_stim;
            end
            output = flodog_normalize_matrix(adapted_responses, ...
                4, ... % normalization window size relative to filter being normed (n)
                1, ... % aspect ratio of window, 1 = round
                .5); % weighted sum across freqs (m)
            patch_lightness = mean(output(stimulus == .5));
            results(a, b, c) = patch_lightness;
        end
    end
end
if saveflag
    save(sprintf(...
        '../data/flodog/%s_%s.mat',...
        stim, adaptor), 'results', 'alphas', 'mus', 'sigmas')
end
end