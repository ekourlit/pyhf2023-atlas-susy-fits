import numpy as np
from typing import List, Dict, Tuple
from cabinetry.model_utils import ModelPrediction
from cabinetry.fit.results_containers import FitResults
from pyhf.pdf import Model
import math
import matplotlib.pyplot as plt
import cabinetry
from pandas import DataFrame, Series
import scipy

def blind_data(data: List, 
               config: Dict,
               model_pred: ModelPrediction,
               model: Model
               ) -> List:
    '''
    Utility function to blind the data in the SRs (defined in config)
    by substituting with floor of total MC (found in model_pred)
    '''

    SRs =[]

    for region in config['Regions']:
        region_name = region['Name']
        
        # strip out _var
        if '_' in region_name:
            region_name = region_name.split('_')[0]

        if 'SR' in region_name: SRs.append(region_name)
    
    # create a copy by-value of data so we don't modify the original list
    local_data = data.copy()

    for region in SRs:
        sr_indices = model.config.channel_slices[region]
        total_mc = np.sum(model_pred.model_yields[sr_indices])
        local_data[sr_indices] = [np.floor(total_mc)]
    
    return local_data

def plot_pulls(model_postfit: ModelPrediction, 
               data: List[float],
               ylim: int = 2,
               xticks_fontsize: int = 16
               ) -> None:
    '''
    Draw the Data/MC(post-fit) pull plot
    '''

    model_yields = model_postfit.model_yields
    model_errors = model_postfit.total_stdev_model_channels
    region_names = model_postfit.model.config.channels.copy()
    regions_to_remove = [region for region in region_names if '_' in region]
    for region in regions_to_remove:
        region_names.remove(region)

    sigmas = []
    for i,region in enumerate(model_yields):
        if i == len(region_names): break

        yields = list(map(lambda x: x[0], region))
        total_SM = sum(yields)
        total_SM_error = model_errors[i][-1]
        datum = data[i] # only use the first len(model_yields) entries
        sigma = calculate_ATLAS_significance(datum, total_SM, total_SM_error)
        sigmas.append(sigma)

    fig = plt.figure(figsize=(7, 4), dpi=150)
    plt.bar(region_names, sigmas)

    # cosmetics
    plt.ylim([-ylim, ylim])
    y_hlines = range(-ylim, ylim+1)
    for y in y_hlines:
        plt.axhline(y=y, linewidth= 1, color='gray', linestyle= 'dashed')
    # plt.ylabel(r'$({n}_{obs} - n_{pred})/\sigma_{tot}$')
    plt.ylabel("Significance")

    # Set x tick font size
    ax = plt.gca()
    for label in (ax.get_xticklabels()):
        label.set_fontsize(xticks_fontsize)

    fig.set_tight_layout(True)

def calculate_ATLAS_significance(nbObs: float, 
                                 nbExp: float, 
                                 nbExpEr: float
                                 ) -> float:
    '''
    Calculate the significance as defined in https://cds.cern.ch/record/2643488
    '''
    factor1 = nbObs*math.log( (nbObs*(nbExp+nbExpEr**2))/(nbExp**2+nbObs*nbExpEr**2) )
    factor2 = (nbExp**2/nbExpEr**2)*math.log( 1 + (nbExpEr**2*(nbObs-nbExp))/(nbExp*(nbExp+nbExpEr**2)) )

    if nbObs < nbExp:
        pull  = -math.sqrt(2*(factor1 - factor2))
    else:
        pull  = math.sqrt(2*(factor1 - factor2))
    
    return pull

def uncertainties_impact_method1(model: Model, 
                                 fit_results: FitResults
                                 ) -> Dict[str, Dict[str, Tuple[float, float]]]:
    '''
    This function calculates the systematic uncertainty breakdown on the post-fit background prediction using the method 1 of the HistFitter paper (https://arxiv.org/pdf/1410.1280.pdf, page 27).
    # VK: this methodology give partial uncertainties higher than the total :/ are you sure post-fit calcelation is taking place?
    
    Returns:
    Dict[region, Dict[NP, Tuple[error, relative_error]]] sorted in descending order
    '''

    # get the channels of the model
    channels = cabinetry.model_utils._filter_channels(model, None)

    # keep the nominal total post-fit prediction per channel handy
    nominal_model_postfit = cabinetry.model_utils.prediction(model, fit_results=fit_results)
    nominal_total_yield_per_channel = [round(np.sum(yield_per_channel), 2) for yield_per_channel in nominal_model_postfit.model_yields]
    
    results = {}
    # loop over the post-fit results
    for unc, label in zip(fit_results.uncertainty, fit_results.labels):
        # print the uncertainty label you are working on
        print(f" estimating the impact of {label} ...")
    
        # set all the parameter uncertainties to zero except for the current parameter
        tmp_unc_array = np.array([0.0 if label_ != label else unc for label_ in fit_results.labels])

        # make a new FitResults object with same properties as fit_results except new uncertainties
        tmp_fit_results = FitResults(fit_results.bestfit,
                                     tmp_unc_array,
                                     fit_results.labels,
                                     fit_results.corr_mat,
                                     fit_results.best_twice_nll,
                                     fit_results.goodness_of_fit)

        # obtain post-fit model prediction
        tmp_model_postfit = cabinetry.model_utils.prediction(model, fit_results=tmp_fit_results)

        # uncertainties per channel
        total_unc_per_channel = [round(unc_per_channel[-1], 2) for unc_per_channel in tmp_model_postfit.total_stdev_model_channels]
        total_rel_unc_per_channel = ( np.round((np.array(total_unc_per_channel)/np.array(nominal_total_yield_per_channel))*100, 1) ).tolist()

        # combine channels and total_unc_per_channel in a dict
        channel_unc_dict = dict(zip(channels, zip(total_unc_per_channel, total_rel_unc_per_channel)))

        results[label] = channel_unc_dict

    # results per channel
    results_per_channel = {}
    for channel in channels:
        results_per_channel[channel] = {unc_label: result[channel] for unc_label, result in results.items()}
        # sort by descending uncertainty
        results_per_channel[channel] = {k: v for k, v in sorted(results_per_channel[channel].items(), key=lambda item: item[1][0], reverse=True)}

    return results_per_channel

# helper function definition
def get_results_df(results: Dict
                   ) -> DataFrame:
    '''
    Convert the results dictionary into a dataframe
    '''

    # create dataframe structure
    results_df = DataFrame(columns=['Sq', 
                                    'Hplus',
                                    'observed_p_value',
                                    'expected_p_value',
                                    'expected_p_value_plusOneSigma',
                                    'expected_p_value_minusOneSigma',
                                    'observed_signif',
                                    'expected_signif',
                                    'expected_signif_plusOneSigma',
                                    'expected_signif_minusOneSigma'])

    # fill dataframe from results dictionary
    for i,key in enumerate(results.keys()):
        stop_m = int(key.split('_')[-2])
        chi1_m = int(key.split('_')[-1])
        obs_p_val = float(results[key].result()[0] if results[key].result()[0] is not None else np.nan)
        exp_p_val = float(results[key].result()[1][1] if results[key].result()[1] is not None else np.nan)
        exp_p_val_plusOneSigma = float(results[key].result()[1][2] if results[key].result()[1] is not None else np.nan)
        exp_p_val_minusOneSigma = float(results[key].result()[1][0] if results[key].result()[1] is not None else np.nan)
        obs_signif = scipy.stats.norm.isf(obs_p_val, 0, 1)
        exp_signif = scipy.stats.norm.isf(exp_p_val, 0, 1)
        exp_signif_plusOneSigma = scipy.stats.norm.isf(exp_p_val_plusOneSigma, 0, 1)
        exp_signif_minusOneSigma = scipy.stats.norm.isf(exp_p_val_minusOneSigma, 0, 1)
        
        results_df.loc[i] = Series({'Sq':stop_m,
                                    'Hplus':chi1_m,
                                    'observed_p_value':obs_p_val, 
                                    'expected_p_value':exp_p_val, 
                                    'expected_p_value_plusOneSigma':exp_p_val_plusOneSigma, 
                                    'expected_p_value_minusOneSigma':exp_p_val_minusOneSigma,
                                    'observed_signif':obs_signif, 
                                    'expected_signif':exp_signif, 
                                    'expected_signif_plusOneSigma':exp_signif_plusOneSigma, 
                                    'expected_signif_minusOneSigma':exp_signif_minusOneSigma})

    # calculate DM column
    results_df['DeltaM'] = results_df['Sq'] - results_df['Hplus']
    # fix types
    results_df = results_df.astype({'Sq':int, 'Hplus':int, 'DeltaM':int})
    # bring DeltaM column after chi1
    results_df = results_df[['Sq', 'Hplus', 'DeltaM', 'observed_p_value', 'expected_p_value', 'expected_p_value_plusOneSigma', 'expected_p_value_minusOneSigma', 'observed_signif', 'expected_signif', 'expected_signif_plusOneSigma', 'expected_signif_minusOneSigma']]

    return results_df

def force_signal_yield_to_one(workspace: Dict, 
                              config: Dict
                              ) -> Dict:
    '''
    Force signal yield in SR to 1.00 and remove stat error
    '''
    # make sure the SR channel is the first channel in the workspace
    sr_channel_name = [ region['Name'] for region in config['Regions'] if 'SR' in region['Name'] ][0]
    sr_channel = workspace['channels'][0]
    if sr_channel['name'] != sr_channel_name:
        raise ValueError("SR channel should be the first channel in the workspace")
    
    # loop over the samples of the SR channel
    for sample in sr_channel['samples']:
        if 'Signal' in sample['name']:
            sample['data'] = [1.0]
            # check the type of the fist modifier
            first_modifier_type = sample['modifiers'][0]['type']
            if first_modifier_type != 'staterror':
                raise AssertionError("First modifier should be the staterror")
            # delete the staterror modifier
            del sample['modifiers'][0]

    return workspace

def plot_UL_scan(obs_UL: np.array,
                 obs_results: List[np.array], 
                 exp_ULs: List[np.array], 
                 exp_results: List[np.array], 
                 scan: np.array
                 ) -> None:
    '''
    Utility function to plot the UL scan
    '''

    # setup the plot
    fig = plt.figure(figsize=(7,5), dpi=150)
    ax = plt.gca()

    # axis range
    plt.xlim((0, scan[-1]))
    plt.ylim((0, 1.2))

    # axis labels
    plt.xlabel(r'$\mu_{\mathrm{sig}}$')
    plt.ylabel(r'$\mathrm{CL}_{s}$')

    # tight layout
    plt.tight_layout()

    # the 0-th element of the results tuple is the observed CLs
    observed = list(map(lambda x: x[0], obs_results))
    plt.plot(scan, observed, label=r'$\mathrm{Observed}\,\mathrm{CL}_{s}$', c = 'black')
    # add vertical line on the observed limit
    plt.axvline(obs_UL, color='red', linestyle='-')
    # add text on the observed limit
    text_offset = 1
    text_size = 11
    plt.text(obs_UL+text_offset, 0.4, f'Observed UL = {obs_UL:.2f}', rotation=90, verticalalignment='bottom', c = 'red', fontsize=text_size)

    # the 1-th element of the results tuple is expected_set, the third element of which is the median
    excepted_median = list(map(lambda x: x[1][2], exp_results))
    plt.plot(scan, excepted_median, linestyle='--', c = 'black', label=r'$\mathrm{Expected}\,\mathrm{CL}_{s}$ - Median')
    # the 1-th element of the results tuple is expected_set, the second element of which is the minus 1 sigma interval
    excepted_minus_1sigma = list(map(lambda x: x[1][1], exp_results))
    # the 1-th element of the results tuple is expected_set, the fourth element of which is the plus 1 sigma interval
    excepted_plus_1sigma = list(map(lambda x: x[1][3], exp_results))
    # fill between the expected +/- 1 sigma intervals
    plt.fill_between(scan, excepted_minus_1sigma, excepted_plus_1sigma, color='lime', alpha=0.5, label=r'$\mathrm{Expected}\,\mathrm{CL}_{s}$ $\pm$ $1\sigma$')

    # add vertical line on the expected limit
    plt.axvline(exp_ULs[2], color='red')
    # add text on the expected limit
    plt.text(exp_ULs[2]+text_offset, 0.4, f'Expected UL = {exp_ULs[2]:.2f}', rotation=90, verticalalignment='bottom', c = 'red', fontsize=text_size)

    # add horizontal line on 0.05
    plt.axhline(0.05, color='grey', linestyle='--')
    # add legend
    plt.legend()
