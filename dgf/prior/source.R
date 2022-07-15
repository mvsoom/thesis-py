setwd("/home/marnix/WRK/thesis/py/")

library(data.table)
library(ggplot2)
library(ggExtra)
theme_set(theme_bw())
theme_update(panel.grid.minor.x = element_blank())

full = as.data.table(read.csv(file = "dgf/prior/source.csv",header = T))

# Remove outlier 10?
#full = full[sample != 10]

full[, `:=`(log_prob_diff = log_prob_q - log_prob_p)]

ggplot(full, aes(log_prob_diff)) +
  geom_histogram()

# Estimate KL divergence with Monte Carlo
KL = full[, .(KL = sum(log_prob_diff)/.N, num_samples = .N), key=.(kernel_name, kernel_M, use_oq, impose_null_integral)]

KL[, `:=`(KL_ban = KL*log10(exp(1)))]

ggplot(KL, aes(kernel_M, KL_ban)) +
  geom_line(aes(color = use_oq, linetype = impose_null_integral)) +
  facet_wrap(vars(kernel_name), labeller = "label_both") +
  scale_x_log10(breaks=unique(full$kernel_M))

KL[, .SD[order(KL_ban)][1], key=.(kernel_name)]

# Check out posterior estimates
post = full[kernel_M == 256 & use_oq == "True" & impose_null_integral == "True" & kernel_name == "Matern32Kernel"]

ggplot(post) +
  geom_histogram(aes(T))
