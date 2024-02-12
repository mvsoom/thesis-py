# General
- yoshii paper: sparse gamma coefficients on a grid can encode (T, OQ) combinations in the basis functions
- can a MVN be weighed (compounded) by another MVN? As in p(w) = int dt MVN(w; f(t))*MVN(t)
- - Because then Laplace approximation on t = theta could act as weights for the posterior MVN on the basis functions coefficients w
- Bayesian Analysis of Arma models (00027.pdf) in mail
- Bayesian Analysis of ARMA Processes: Complete Sampling Based Inference
Under Full Likelihoods: reparametrization for stationaroty
- - Prior repartitioning with Uniform prior for AR params over hypercube; and: MVN prior for AR params derived from ML estimate from ordinary AR algorithm? Via Laplace. The second thing is done in this paper
- - can use Beta distributions to parametrize AR coeff in hypercybe domain and then update with VI as in yoshii
- what did jaynes say about all this?
- And the art of computing (the bible for algorithms) about AR?
- Learning from previous priors when increasing p with the russian guys NS prior repartitioning thing? (Maxent conf)
- Only do p inferemce and keep it constant over all pitch periods: is good
- - Use a periodic kernel after all? Can this fit into dbltrf() idea? in lieu of correlated amplitudes
- - our formant prior lengthscale quantifies how many pitch periods approximately the VT stays stationary (dozens?)
- We can integrate out sigma (temperature, smoothness) analytically as in our entropy paper
- ARMA might be necessary after all because we care about phase (the GF cycle) and not only power


- Does exactly the same as us: GP_Sem4_att1.pdf (in ~/WRK)
- - To get more important examples, we decompose the u(t)
 curves into interesting constituents according to our GP model


## Papers

- @Yoshii2013
  - Introduces (AP, nonvoicedness, enumerative kernel structure) into GP/VI framework
  - No followup in the literature
  - NS/VI inference

- @Kleibergen2000
  - How to specify an ARMA process from an AR model
  - Does not seem practical

# Positivity of gf
- - We already have a mechanism for this: the flow squared! this is probably the way to go
- - other interesting papers are 
- - Chai2019 (https://arxiv.org/pdf/1802.04782.pdf): using sqrt or log map and second order approximations -- seems to work quite fine
- - - Spiller2023: (gf(t) ~ max(0, GP))
- - - Swiler2020: general survey
- - called "shape constrained GPs" or "linear inequality constrained GPs"
- - in general: i did not find a good approach. it comes down to the expression for the log likelihood of a truncated multivariate normal (and sampling from it), and i can't find a good solution for this. Other approaches comes from constrained convex optimization (Nonnegativity-Enforced Gaussian Process Regression)
- - https://arxiv.org/pdf/1901.03134.pdf (Agrell2019) states that (Sec 3.3) the correction for positivity has little effect on the log likelihood in the large data regime, so making the gf positive is simply done as an after processing step... could always do this ... many other papers also assume the hyperparams to be known: "This explains that many papers [see, for instance, Agrell (2019) and Veiga and Marrel (2020)] propose focusing on the
unconstrained log likelihood only." (p. 64,  Perrin & Da Vega 2019)