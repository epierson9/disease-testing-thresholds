data {
  int<lower=1> N; // number of observations
  int<lower=1> R; // number of races
  int<lower=1> D; // number of counties
  
  int<lower=1,upper=R> r[N]; // race 
  int<lower=1,upper=D> d[N]; // county
  
  int<lower=0> n[N]; // county population
    int<lower=0> s[N]; // # of tests
int<lower=0> h[N]; // # of positive tests (hits)
}

parameters {	
  // hyperparameters
  real<lower=0> sigma_t; #standard deviation for the normal the thresholds are drawn from. 
  
  // test thresholds
  vector[N] t_i;
  
  // parameters for signal distribution
  vector[R] phi_r;
  vector[D-1] phi_d_raw;
  real mu_phi;
  
  real mu_delta;
}

transformed parameters {
  vector[D] phi_d;
  vector[N] phi;
  vector[N] delta;
  vector<lower=0, upper=1>[N] search_rate;
  vector<lower=0, upper=1>[N] hit_rate;
  real successful_search_rate;
  real unsuccessful_search_rate;

  phi_d[1]      <- 0;
  phi_d[2:D]    <- phi_d_raw;

  
  for (i in 1:N) {	
    // phi is the fraction of people of race r, d who are sick. 
    phi[i]    <- inv_logit(phi_r[r[i]] + phi_d[d[i]]);
    
    // mu is the center of the signal distribution for sick people. 
    // note slightly different parameterization - we have a single mu across all races/locations. 
    // and have more heterogeneity in phi. Thought is that most of the heterogeneity is in fraction of race/location which is sick. 
    delta[i] <- exp(mu_delta);
    
    successful_search_rate <- phi[i] * (1 - normal_cdf(t_i[i], delta[i], 1));
    unsuccessful_search_rate <- (1 - phi[i]) * (1 - normal_cdf(t_i[i], 0, 1)); 
    search_rate[i] = (successful_search_rate + unsuccessful_search_rate);
    hit_rate[i] = successful_search_rate / search_rate[i];
   }
}

model {  
  // Draw threshold hyperparameters
  sigma_t ~ normal(0, 1);
  
  // Draw race parameters. Each is centered at a mu, and we allow for inter-race heterogeneity. 
  mu_phi ~ normal(-3, 1);
  mu_delta ~ normal(0, 1);
  
  phi_r    ~ normal(mu_phi, 1);
  
  // Draw department parameters (for un-pinned departments)
  phi_d_raw    ~ normal(0, 1);   // we have wider prior for phi_d than in the original threshold test, because locations may vary a lot. 
  //thresholds
  t_i ~ normal(0, sigma_t);

  s ~ binomial(n, search_rate);
  h ~ binomial(s, hit_rate);
}

generated quantities {}
