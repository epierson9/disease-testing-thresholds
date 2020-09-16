import json
import pandas as pd
import numpy as np
import pystan
from scipy.stats import norm, binom, spearmanr, pearsonr
import random
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import time

def get_acs_data(grouping):
    # Read in population by zipcode / county. 
    # 2018 5-year ACS
    # From README: 
    # B04001_003:         White Alone
    # B04001_004:         Black or African American Alone
    #B04001_010:      Hispanic or Latino
    assert grouping in ['zipcode', 'county']
    race_cols = ['SE_B04001_003', 'SE_B04001_004', 'SE_B04001_010']
    new_names = ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic/Latino']
    if grouping == 'zipcode':
        race_by_location = pd.read_csv('acs_2018_5_year_census_data_zipcode.csv')
        assert race_by_location['Geo_NAME'].duplicated().sum() == 0
        race_by_location['location'] = race_by_location['Geo_NAME'].map(lambda x:'Zip Code ' + x.split()[1].strip())
    else:
        race_by_location = pd.read_csv('acs_2018_5_year_census_data_county.csv')
        assert race_by_location['Geo_FIPS'].duplicated().sum() == 0
        race_by_location['location'] = race_by_location['Geo_FIPS'].map(lambda x:'County %05d' % x)
    assert race_by_location['location'].duplicated().sum() == 0
    race_by_location = race_by_location[['location'] + race_cols]
    race_by_location.columns = ['location'] + new_names
    race_by_location = pd.melt(race_by_location, 
            value_vars=['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic/Latino'], 
            id_vars=['location'], 
            value_name='Population', 
            var_name='race_group')
    return race_by_location

def load_indiana_data(indiana_data_filename, method):
    """
    Spot-checked that this agrees well with Indiana website. 

    Note that TESTS refers to unique individuals tested [not total tests administered]. 

    # Spot check against Indiana data on website: county 18141 on 8/16: 
    # 3,844 positive cases, 50,201 tests
    # cases: 34.3% white, 15.1 black, 6.9 hispanic [this is not renormalizing for unknowns]
    # tests: 46.4% white, 11.1% black, 2.6 hispanic [this is not renormalizing for unknowns]

    # so this means cases: 1318 white, 580 black, 265 hispanic. 
    # tests: 23293 white, 5572 black, 1305 hispanic. 

    # this matches up well with our processed data. Differences are probably due either to slight differences in when
    # data was updated or that their percentages are rounded. 

    # spot check against census data for populations. Per http://www.stats.indiana.edu/profiles/profiles.asp?scope_choice=a&county_changer=18141, 
    # County 18141 has Black population of 37k, white population of 216k (I think these numbers both include Hispanic), hispanic population of 24812. 
    # Similarly, https://datausa.io/profile/geo/st.-joseph-county-in#demographics reports 195k white non-hispanic, 35k black non-hispanic. this all lines up well. 

    # Comparison to overall indiana data from website. 
    # total tests 906,851, total cases 80,415
    # tests 60.1 white, 8.2 black, 3.5% hispanic: 545017 white, 74361 black, 31739 hispanic. Good agreement. 
    # cases: 49.3 white, 11.1% black, 10% hispanic. 39644 white, 8926 black, 8041 hispanic. Good agreement. 

    # confirmed that covid tracking project rates of MISSING data line up well with Indiana's numbers for both race/ethnicity and positive/negative tests. However, Atlantic's data on known hispanic negative tests in COVID race tracking data seems too low. 
    from Indiana website, Known Hispanic is 0.035 * 906,851 tests = 31739 tests and 
    10% of 80,415 cases = 8041 cases. So 8041 positive tests and 23698 negative tests as opposed to our parsed Atlantic data which is 7323 positive tests and 3456 negative tests. I do not think it is that we are parsing the data wrong - I think it might be that the Atlantic's data has the wrong data from the Indiana website. Wrote to Atlantic about this. 

    Unknown data fractions match up as well. 
    """ 
    indiana_data = json.load(open(indiana_data_filename))
    indiana_d = []
    assert method in ['reweight_by_hispanic_and_unknown', 
                      'subtract_hispanic_from_white',
                      'just_use_raw_numbers'] # Original data is aggregated by race and separately by ethnicity. Threshold test expects both combined. We have to make some assumptions to do this, so we try doing it three different ways. 

    total_cases = 0
    total_tests = 0
    missing_data = []

    for county in indiana_data['objects']['cb_2015_indiana_county_20m']['geometries']:
        race_data = pd.DataFrame(county['properties']['VIZ_D_RACE']).set_index('RACE')
        ethnicity_data = pd.DataFrame(county['properties']['VIZ_D_ETHNICITY']).set_index('ETHNICITY')
        assert race_data['COVID_COUNT'].sum() == ethnicity_data['COVID_COUNT'].sum() == county['properties']['COVID_COUNT'] # race and ethnicity cases sum to the same thing.
        assert race_data['COVID_TEST'].sum() == ethnicity_data['COVID_TEST'].sum() == county['properties']['COVID_TEST'] 
        total_cases += county['properties']['COVID_COUNT']
        total_tests += county['properties']['COVID_TEST']
        
        county_fips = 'County ' + county['properties']['STATEFP'] + county['properties']['COUNTYFP'] 

        # lots of missing ethnicity data, so some of our methods compute the fraction of those of known ethnicity who are Hispanic. 
        hisp_test_frac = 1.*ethnicity_data.loc['Hispanic or Latino', 'COVID_TEST'] / (
            ethnicity_data.loc['Hispanic or Latino', 'COVID_TEST'] + 
            ethnicity_data.loc['Not Hispanic or Latino', 'COVID_TEST'])
        hisp_case_frac = 1.*ethnicity_data.loc['Hispanic or Latino', 'COVID_COUNT'] / (
            ethnicity_data.loc['Hispanic or Latino','COVID_COUNT'] + 
            ethnicity_data.loc['Not Hispanic or Latino', 'COVID_COUNT'])

        known_race_test_frac = 1 - 1.*race_data.loc['Unknown', 'COVID_TEST']/race_data['COVID_TEST'].sum()
        known_race_case_frac = 1 - 1.*race_data.loc['Unknown', 'COVID_COUNT']/race_data['COVID_COUNT'].sum()
        known_ethnicity_test_frac =  1 - 1.*ethnicity_data.loc['Unknown', 'COVID_TEST']/ethnicity_data['COVID_TEST'].sum()
        known_ethnicity_case_frac =  1 - 1.*ethnicity_data.loc['Unknown', 'COVID_COUNT']/ethnicity_data['COVID_COUNT'].sum()
        missing_data.append({'test_race_known':known_race_test_frac, 
            'county':county_fips, 
            'case_race_known':known_race_case_frac, 
            'test_eth_known':known_ethnicity_test_frac, 
            'case_eth_known':known_ethnicity_case_frac})
        for race_group in ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic/Latino']:
            if method == 'reweight_by_hispanic_and_unknown':
                # basically, we assume that data is missing at random and that race/ethnicity are independent. 
                if race_group == 'Non-Hispanic Black':
                    if 'Black or African American' not in race_data.index:
                        case_count = 0
                        test_count = 0
                    else:
                        case_count = int(race_data.loc['Black or African American', 'COVID_COUNT'] * (1 - hisp_case_frac) / known_race_case_frac)
                        test_count = int(race_data.loc['Black or African American', 'COVID_TEST'] * (1 - hisp_test_frac) / known_race_test_frac)
                elif race_group == 'Non-Hispanic White':
                    case_count = int(race_data.loc['White','COVID_COUNT'] * (1 - hisp_case_frac) / known_race_case_frac)
                    test_count = int(race_data.loc['White', 'COVID_TEST'] * (1 - hisp_test_frac) / known_race_test_frac)
                else:
                    case_count = int(hisp_case_frac * county['properties']['COVID_COUNT'])
                    test_count = int(hisp_test_frac * county['properties']['COVID_TEST'])   
            elif method == 'just_use_raw_numbers':
                # literally just take the raw counts directly from the data. Probably not the best method, because makes no attempt to account for missing data. 
                # Also doesn't consider overlap between race and ethnicity. 
                if race_group == 'Non-Hispanic Black':
                    if 'Black or African American' not in race_data.index:
                        case_count = 0
                        test_count = 0
                    else:
                        case_count = int(race_data.loc['Black or African American', 'COVID_COUNT'])
                        test_count = int(race_data.loc['Black or African American', 'COVID_TEST'])
                elif race_group == 'Non-Hispanic White':
                    case_count = int(race_data.loc['White','COVID_COUNT'])
                    test_count = int(race_data.loc['White', 'COVID_TEST'])
                else: 
                    case_count = int(ethnicity_data.loc['Hispanic or Latino', 'COVID_COUNT'])
                    test_count = int(ethnicity_data.loc['Hispanic or Latino', 'COVID_TEST'])
            elif method == 'subtract_hispanic_from_white':
                # Also very imperfect. Just subtract Hispanic count from white count. 
                if race_group == 'Non-Hispanic Black':
                    if 'Black or African American' not in race_data.index:
                        case_count = 0
                        test_count = 0
                    else:
                        case_count = int(race_data.loc['Black or African American', 'COVID_COUNT'])
                        test_count = int(race_data.loc['Black or African American', 'COVID_TEST'])
                elif race_group == 'Non-Hispanic White':
                    case_count = int(race_data.loc['White','COVID_COUNT']) - int(ethnicity_data.loc['Hispanic or Latino', 'COVID_COUNT'])
                    test_count = int(race_data.loc['White', 'COVID_TEST']) - int(ethnicity_data.loc['Hispanic or Latino', 'COVID_TEST'])
                else:
                    case_count = int(ethnicity_data.loc['Hispanic or Latino', 'COVID_COUNT'])
                    test_count = int(ethnicity_data.loc['Hispanic or Latino', 'COVID_TEST'])
            else:
                raise Exception("Invalid method")
            assert test_count >= case_count
            if test_count < 0:
                print("WARNING: test count is %i. Thresholding at 0" % test_count)
                test_count = 0
            if case_count < 0:
                print("WARNING: case count is %i. Thresholding at 0" % case_count)
                case_count = 0


            indiana_d.append({'location':county_fips, 
                             'race_group':race_group,
                             'tests':test_count, 
                              'cases':case_count})
            
        
    missing_data = pd.DataFrame(missing_data)
    print("Statistics on missing data")
    print(missing_data.describe())
    indiana_d = pd.DataFrame(indiana_d)    
    old_len = len(indiana_d)
    assert pd.isnull(indiana_d).values.sum() == 0
    print("Number of counties: %i; %i rows; total cases: %i; total INDIVIDUALS TESTED %i" % (
        len(set(indiana_d['location'])), 
        len(indiana_d), 
        total_cases,
        total_tests)) # checks out - The U.S. state of Indiana has 92 counties. Case and test counts also match website.
    assert len(indiana_d) == (len(set(indiana_d['location'])) * len(set(indiana_d['race_group'])))
    assert indiana_d[['location', 'race_group']].duplicated().sum() == 0

    county_acs_data = get_acs_data(grouping='county')
    indiana_d = pd.merge(indiana_d, county_acs_data, how='inner', on=['location', 'race_group'])
    assert pd.isnull(indiana_d).values.sum() == 0
    assert len(indiana_d) == old_len
    print(indiana_d.describe().transpose())

    return indiana_d

def make_rate_plot(white, black, hispanic, rate_name, figname=None, rate_name_for_axis_label=None):
    """
    Scatterplot with white on x-axis, Black/Hispanic on y-axis. Accomodates missing data. 
    """
    if rate_name_for_axis_label is None:
        rate_name_for_axis_label = rate_name
    white_nan = np.isnan(white)
    black_nan = np.isnan(black)
    hispanic_nan = np.isnan(hispanic)
    max_val = max(list(white[~white_nan]) + list(black[~black_nan]) + list(hispanic[~hispanic_nan])) + 0.05
    plt.figure(figsize=[8, 4])
    plt.subplot(1, 2, 1) # white + Black
    good_idxs = ~(white_nan | black_nan)
    plt.scatter(white[good_idxs], black[good_idxs])
    print("For %s, %i/%i points with data have larger values for white than Black" % 
          (rate_name, (white[good_idxs] > black[good_idxs]).sum(), good_idxs.sum()))
    plt.plot([0, max_val], [0, max_val], linestyle='--', color='black')
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.xlabel("White %s" % rate_name_for_axis_label)
    plt.ylabel("Black %s" % rate_name_for_axis_label)
    
    plt.subplot(1, 2, 2) # white + Hispanic
    good_idxs = ~(white_nan | hispanic_nan)
    plt.scatter(white[good_idxs], hispanic[good_idxs])
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.plot([0, max_val], [0, max_val], linestyle='--', color='black')
    plt.xlabel("White %s" % rate_name_for_axis_label)
    plt.ylabel("Hispanic %s" % rate_name_for_axis_label)
    print("For %s, %i/%i points with data have larger values for white than Hispanic" % 
          (rate_name, (white[good_idxs] > hispanic[good_idxs]).sum(), good_idxs.sum()))
    plt.subplots_adjust(wspace=.3)
    if figname is not None:
        plt.savefig(figname)
    plt.show()

def filter_and_annotate_raw_data(results, save_figures, min_race_group_frac, min_race_group_n):
    """
    Make basic descriptive plots, and filter for counties with at least a fraction min_race_group_frac of all race_groups and min_race_group_n of all race_groups. 
    """
    assert min_race_group_frac >= 0 and min_race_group_frac <= 1
    assert min_race_group_n >= 0
    results = results.copy()
    original_len = len(results)
    n_original_locs = len(set(results['location']))

    total_location_pops = results.groupby('location')['Population'].sum().reset_index()
    total_location_pops.columns = ['location', 'total_location_pop']
    results = pd.merge(results, total_location_pops, on='location', how='inner')
    results['fraction_of_location_pop'] = results['Population'] / results['total_location_pop']

    total_tests_for_location = results.groupby('location')['tests'].sum().reset_index()
    total_tests_for_location.columns = ['location', 'total_tests_for_location']
    results = pd.merge(results, total_tests_for_location, on='location', how='inner')
    results['relative_test_probability'] = results['tests'] / results['total_tests_for_location']
    assert len(results) == original_len

    # plots of population fractions
    print(results.groupby('race_group')[['relative_test_probability', 'fraction_of_location_pop']]
          .describe()
          .transpose())

    locs_with_at_least_some_tests_for_all_groups = (results
                                                    .loc[(results['Population'] >= min_race_group_n) &  
                                                         (results['fraction_of_location_pop'] >= min_race_group_frac)]
                                                    .groupby('location').size())
    good_locs = set((locs_with_at_least_some_tests_for_all_groups[
        locs_with_at_least_some_tests_for_all_groups == locs_with_at_least_some_tests_for_all_groups.max()].index))

    original_counts = results.groupby('race_group')[['tests', 'Population', 'cases']].sum()
    results = results.loc[results['location'].map(lambda x:x in good_locs)]
    print("After filtering for locations, %i/%i locations remain" % 
          (len(set(results['location'])), n_original_locs))
    new_counts = results.groupby('race_group')[['tests', 'Population', 'cases']].sum()

    print("Ratio of cases, pop, and tests retained by race/ethnicity group")
    print(new_counts/original_counts)

    # Cases   
    results['cases_per_pop'] = results['cases'] / results['Population']
    per_capita_cases = pd.pivot_table(results[['race_group', 'location', 'cases_per_pop']], 
                   columns='race_group', 
                   index='location').reset_index()

    make_rate_plot(white=per_capita_cases[('cases_per_pop', 'Non-Hispanic White')].values, 
                black=per_capita_cases[('cases_per_pop', 'Non-Hispanic Black')].values, 
                hispanic=per_capita_cases[('cases_per_pop', 'Hispanic/Latino')].values, 
                rate_name='cases per pop')

    # tests per pop
    results['tests_per_pop'] = results['tests'] / results['Population']
    per_capita_tests = pd.pivot_table(results[['race_group', 'location', 'tests_per_pop']], 
                   columns='race_group', 
                   index='location').reset_index()

    make_rate_plot(white=per_capita_tests[('tests_per_pop', 'Non-Hispanic White')].values, 
                black=per_capita_tests[('tests_per_pop', 'Non-Hispanic Black')].values, 
                hispanic=per_capita_tests[('tests_per_pop', 'Hispanic/Latino')].values, 
                rate_name='tests per pop')

    # cases per test
    results['pos_frac'] = results['cases'] / results['tests']
    hit_rates_by_race = pd.pivot_table(results[['race_group', 'location', 'pos_frac']], 
                   columns='race_group', 
                   index='location').reset_index()

    make_rate_plot(white=hit_rates_by_race[('pos_frac', 'Non-Hispanic White')].values, 
                black=hit_rates_by_race[('pos_frac', 'Non-Hispanic Black')].values, 
                hispanic=hit_rates_by_race[('pos_frac', 'Hispanic/Latino')].values, 
                rate_name='positive test frac', 
                rate_name_for_axis_label='positivity rate',
                figname='indiana_positivity_rate.pdf' if save_figures else None)

    results = results.sort_values(by='location') # this is what multinomial code expects. 
    return results

def draw_from_signal_distribution(n, phi, delta, sigma_g=1):
    # Draws samples from the x distribution (so this is in signal space, not probability space). 
    n_positive = np.random.binomial(n=n, p=phi)
    n_negative = n - n_positive
    signal_samples = (list(np.random.normal(size=n_positive, loc=delta, scale=sigma_g)) + 
                    list(np.random.normal(size=n_negative, loc=0, scale=1)))
    return pd.DataFrame({"signal_samples":signal_samples, 
                         "has_covid":[0] * n_negative + [1] * n_positive})

def fit_model(model_filename, results, use_multinomial_model):
    """
    Actually fit the model. This will work with the multinomial model discussed in the AISTATS paper even though we did not end up using it
    because it makes less sense for COVID data. 
    """
    unique_race_groups = sorted(list(set(results['race_group'])))
    unique_locations = sorted(list(set(results['location'])))
    assert len(results) == len(unique_race_groups) * len(unique_locations)
    n_chains = 4
    n_iter = 10000
    assert model_filename in ['fast_multinomial_model_no_deltas.stan', 
                              'poisson_mixture_model_no_deltas.stan', 
                              'binomial_mixture_model_no_deltas.stan']

    stan_data = {'N': len(results),
                 'R':len(unique_race_groups),
                 'D':len(unique_locations),
                 'r':results['race_group'].map(lambda x:unique_race_groups.index(x) + 1).values, 
                 'd':results['location'].map(lambda x:unique_locations.index(x) + 1).values,
                 's':results['tests'].astype(int).values, 
                 'h':results['cases'].astype(int).values}
    if use_multinomial_model: # population enters in differently in multinomial model. 
        stan_data['base_population_proportions'] = results['fraction_of_location_pop'].values
    else:
        stan_data['n'] = results['Population'].astype(int)

    model = pystan.StanModel(model_filename)
    print("Beginning model fitting using model %s with %i locations and %i race groups, %i rows total" % 
        (model_filename, stan_data['D'], stan_data['R'], stan_data['N']))
    t0 = time.time()
    fit = model.sampling(data=stan_data, iter=n_iter, chains=n_chains)
    print("Finished fitting model in %2.3f seconds" % (time.time() - t0))
    return fit

def signal_to_p(x, phi, delta, sigma_g=1):
    # Converts x -> p. Translated right out of the original R code. 
    p = phi * norm.pdf(x, loc=delta, scale=sigma_g) / (phi * norm.pdf(x, loc=delta, scale=sigma_g) + (1 - phi) * norm.pdf(x, loc=0, scale=1))
    alternate_p = 1/(1 + np.exp(-delta * x + delta ** 2 / 2) * (1 - phi)/phi) # expression used in paper - should be the same. 
    assert np.allclose(p, alternate_p)
    return p

def analyze_fitted_model(fit, results, use_multinomial_model, save_figures=False):
    """
    Do a bunch of analyses on the fitted model - not the best code design. 
    """
    results = results.copy()

    # print basic measures of fit quality. 
    summary = fit.summary()
    rhat_col_idx = summary['summary_colnames'].index('Rhat')
    rhats = summary['summary'][:, rhat_col_idx]

    rhat_nan = np.isnan(rhats)
    print("Rhat ranges from %2.5f-%2.5f with %i Nans in %s" % 
          (np.min(rhats[~rhat_nan]), np.max(rhats[~rhat_nan]), rhat_nan.sum(), ','.join(summary['summary_rownames'][rhat_nan])))
    print(fit.stansummary(pars=['sigma_t', 'mu_delta', 'phi_r']))
    assert np.max(rhats[~rhat_nan]) <= 1.05

    # extract samples. 
    samples = fit.extract(permuted=True) 
    t_i = samples['t_i']
    phi = samples['phi']
    delta = samples['delta']
    assert t_i.shape == phi.shape == delta.shape
    
    # plot thresholds
    thresholds = np.zeros(delta.shape)

    for j in range(t_i.shape[1]):
        thresholds[:, j] = signal_to_p(x=t_i[:, j], phi=phi[:, j], delta=delta[:, j], sigma_g=1)
    results['mean_threshold'] = thresholds.mean(axis=0)
    #results['upper_CI'] = np.percentile(thresholds, axis=0, q=97.5)
    #results['lower_CI'] = np.percentile(thresholds, axis=0, q=2.5)
    #results['CI_width'] = results['upper_CI'] - results['lower_CI']
    #print("CIs on individual thresholds - note that these are not population-weighted, so don't take too literally")
    #print(results.groupby('race_group')[['mean_threshold', 'CI_width']].agg(['median', 'mean']))
    threshold_df = pd.pivot_table(results[['race_group', 'location', 'mean_threshold']], 
               columns='race_group', 
               index='location').reset_index()
    make_rate_plot(white=threshold_df[('mean_threshold', 'Non-Hispanic White')].values, 
                black=threshold_df[('mean_threshold', 'Non-Hispanic Black')].values, 
                hispanic=threshold_df[('mean_threshold', 'Hispanic/Latino')].values, 
                rate_name='threshold', 
                figname='indiana_thresholds.pdf' if save_figures else None)
    plt.show()

    # Risk distributions. 
    all_draws = []
    for i in range(len(results)):
        race_i = results.iloc[i]['race_group']
        phi_i = phi[:, i].mean()
        delta_i = delta[:, i].mean()
        n_i = int(results.iloc[i]['total_tests_for_location'])
        x_draws = draw_from_signal_distribution(n=n_i, phi=phi_i, delta=delta_i)
        assert len(x_draws) == n_i
        p_draws = signal_to_p(x_draws['signal_samples'].values, 
                              phi=phi_i,
                              delta=delta_i,
                              sigma_g=1)
        all_draws.append(pd.DataFrame({'p':p_draws, 'race_group':race_i}))


    all_draws = pd.concat(all_draws)
    upper_lim = .5
    fig = plt.figure(figsize=[8, 4])
    ax = fig.add_subplot(1, 1, 1)
    aggregate_thresholds = {}
    order_to_plot_races = ['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic/Latino']

    for race in order_to_plot_races:
        draws_for_race = all_draws.loc[all_draws['race_group'] == race, 'p'].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.kdeplot(draws_for_race, label=race, gridsize=1000, ax=ax)
        if (draws_for_race > upper_lim).mean() > .05:
            print("WARNING: a lot of mass at the upper tail")
            print((draws_for_race > upper_lim).mean(), race)
        location_weights = 1.*results.loc[results['race_group'] == race, 'total_tests_for_location'].values
        location_weights = location_weights/location_weights.sum()
        assert np.allclose(location_weights.sum(), 1)
        aggregate_threshold = (results.loc[results['race_group'] == race, 'mean_threshold'] * location_weights).sum() 
        
        all_samples_for_aggregate_threshold = np.dot(thresholds[:, (results['race_group'] == race).values], location_weights)
        aggregate_thresholds[race] = {'mean':aggregate_threshold, 'all_draws':all_samples_for_aggregate_threshold}
        assert np.allclose(all_samples_for_aggregate_threshold.mean(), aggregate_threshold)

    ylimits = list(ax.get_ylim())
    ylimits[0] = 0

    
    ax.set_yticks([])
    ax.set_xlim([0, upper_lim])
    for idx, k in enumerate(order_to_plot_races):
        ax.plot([aggregate_thresholds[k]['mean'], aggregate_thresholds[k]['mean']], ylimits, color=sns.color_palette()[idx], linestyle='--', label='%s threshold' % k)
    ax.set_ylim(ylimits)
    ax.set_xlabel("")
    ax.set_xlabel("Risk")
    ax.legend()
    if save_figures:
        plt.savefig("indiana_risk_distributions.pdf")
    plt.show()

    # PPCs
    results['predicted_hit_rate'] = samples['hit_rate'].mean(axis=0)
    
    # search rate depends on whether using multinomial model. 
    if use_multinomial_model:
        results['theta'] = samples['theta'].mean(axis=0)
        theta_sum = results.groupby('location')['theta'].sum().reset_index()
        theta_sum.columns = ['location', 'theta_sum']
        results = pd.merge(results, theta_sum, on='location', how='inner')
        results['predicted_relative_test_probability'] = results['theta'] / results['theta_sum']
        sns.scatterplot(results['relative_test_probability'], 
                    results['relative_test_probability'] - results['predicted_relative_test_probability'], 
                        hue=results['race_group'].values)
        plt.ylim([-.05, .05])
        plt.xlim([0, max(results['relative_test_probability']) + .03])
        plt.plot([0, max(results['relative_test_probability']) + .03], [0, 0], linestyle='--', color='black')
        plt.xlabel("True fraction of tests")
        plt.ylabel("Prediction error")
        plt.show()
    else:
        results['predicted_search_rate'] = samples['search_rate'].mean(axis=0)
        sns.scatterplot(results['tests_per_pop'], 
                    results['tests_per_pop'] - results['predicted_search_rate'], 
                        hue=results['race_group'].values, 
                        s=results['Population'] * .001)
        plt.ylim([-.05, .05])
        plt.xlim([0, max(results['tests_per_pop']) + .03])
        plt.plot([0, max(results['tests_per_pop']) + .03], [0, 0], linestyle='--', color='black')
        plt.xlabel("True tests per pop")
        plt.ylabel("Prediction error")
        if save_figures:
            plt.savefig("indiana_search_rate_ppc.pdf")
        plt.show()

    # hit rate PPC
    sns.scatterplot(results['pos_frac'], 
                results['pos_frac'] - results['predicted_hit_rate'], 
                    hue=results['race_group'].values, 
                    s=results['tests'] * .01)
    max_err = np.abs(results['pos_frac'] - results['predicted_hit_rate']).max()
    plt.ylim([-max_err * 1.5, max_err * 1.5])
    plt.xlim([0, max(results['pos_frac']) + .03])
    plt.plot([0, max(results['pos_frac']) + .03], [0, 0], linestyle='--', color='black')
    plt.xlabel("True positivity rate")
    plt.ylabel("Prediction error")
    if save_figures:
        plt.savefig("indiana_hit_rate_ppc.pdf")
    plt.show()

    # print error rates by race group. 
    results['hit_rate_err'] = np.abs(results['pos_frac'] - results['predicted_hit_rate'])
    results['relative_hit_rate_err']  = results['hit_rate_err'] /results['predicted_hit_rate']

    if use_multinomial_model:
        results['test_prob_err'] = np.abs(results['relative_test_probability'] - results['predicted_relative_test_probability'])
        print(results.groupby('race_group')[['hit_rate_err', 'relative_hit_rate_err', 'test_prob_err']].mean())
    else:
        results['search_rate_err'] = np.abs(results['tests_per_pop'] - results['predicted_search_rate'])
        print(results.groupby('race_group')[['hit_rate_err', 'relative_hit_rate_err', 'search_rate_err']].mean())

    print("\n\n\nDo counties which are more white have lower thresholds and positivity rates?")
    for var in ['mean_threshold', 'pos_frac']:
        for race_group in sorted(list(set(results['race_group']))):
            race_df = results.loc[results['race_group'] == race_group, ['location', var]]
            race_df = race_df.merge(results.loc[results['race_group'] == 'Non-Hispanic White', ['location', 'fraction_of_location_pop']])
            print("Corr between county_frac_white and %s %s: spearman %2.3f; pearsonr %2.3f" % 
              (race_group, 
               var,
               spearmanr(race_df['fraction_of_location_pop'], 
                        race_df[var])[0], 
              pearsonr(race_df['fraction_of_location_pop'], 
                        race_df[var])[0]))

        average_by_county = results.groupby('location')[var].mean().reset_index()
        average_by_county = average_by_county.merge(
            results.loc[results['race_group'] == 'Non-Hispanic White', ['location', 'fraction_of_location_pop']])

        print("Corr between county_frac_white and average %s across ALL groups: spearman %2.3f; pearsonr %2.3f" % 
              (var, 
               spearmanr(average_by_county['fraction_of_location_pop'], 
                        average_by_county[var])[0], 
              pearsonr(average_by_county['fraction_of_location_pop'], 
                        average_by_county[var])[0]))

    print("Aggregate thresholds")
    for k in aggregate_thresholds:
        print('%-50s %2.3f (%2.3f, %2.3f)' % 
            (k, 
            aggregate_thresholds[k]['mean'], 
            np.percentile(aggregate_thresholds[k]['all_draws'], 2.5),
            np.percentile(aggregate_thresholds[k]['all_draws'], 97.5)))
    return aggregate_thresholds


