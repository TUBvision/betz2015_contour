% INPUT: filter_response - a structure of filter responses, stdev_pixels,
%                and orientations from BM_filter.m
%      
% OUTPUT: modelOut - the output of the FLODOG model with local normalization
%                   by orientation and freq, using sdmix to control
%                   influence of nearby freqs. 
% (c) 2007 Alan Robinson, Paul Hammon, Virgina de Sa.

function modelOut = flodog_normalize_matrix(filter_response, sigx, sr,  sdmix)


add_const = 10^(-6);  

%disp('weighting and normalizing');

BM_model_params

%disp('weight filter responses');

% loop over the orientations
for o = 1 : length(orientations)

    this_norm = 0;

    % loop over spatial frequencies
    for f = 1 : length(stdev_pixels)

        % get the filtered response
        filt_img = squeeze(filter_response(o, f, :, :));

        % weight each response
                
        filter_response(o, f, :, :) = filt_img * w_val(f);
    end
end

% next do local norming


%disp('normalize each filter response');
modelOut = 0;

% loop over the orientations
for o = 1 : length(orientations)

    % loop over spatial frequencies
    for f = 1 : length(stdev_pixels)
        

        filter_resp = squeeze(filter_response(o, f, :, :));

        
        % build a weighted normalizer
        normalizer = 0;
        area = 0;
        for (wf = 1: length(stdev_pixels))
            gweight = gauss(f-wf, sdmix); 
            area = area + gweight;
            normalizer = normalizer + (squeeze(filter_response(o, f, :, :)) .* gweight);
        end
        normalizer = normalizer ./area;
        
        % square
        normalizer_sqr = normalizer .^ 2;

        % create the filter:
        % extent along direction of filter - funciton of frequency
        sig1 = sigx * stdev_pixels(f);

        % perpendicular to filter
        sig2 = sig1 * sr;

        % directed along main axis of filter
        rot = orientations(o) * pi/180;

        % create a unit volume gaussian for filtering
        mask = d2gauss(model_x, sig1, model_y, sig2, rot);
        mask = mask ./ sum(mask(:));

        % filter the image (using unit-sum mask --> mean)
        local_normalizer = ourconv(normalizer_sqr, mask,0);

        % make sure there are no negative numbers due to FFT
        local_normalizer = local_normalizer + 10^(-6);

        % take the square root
        local_normalizer = local_normalizer .^ 0.5;

        % divide through by normalized image
        temp = filter_resp ./ (local_normalizer + add_const);
        
         % weight and add over filt.stdev_pixels
        modelOut = modelOut  + temp;
        clear temp filter_resp normalizer local_normalizer normalizer_sqr;
        %disp('Normalized 1 filter resp...'); 
    end
end


