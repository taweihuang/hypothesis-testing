import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# 1.1 Perform one-sample t test

np.random.seed(1234)
mu = 130
sigma = 15
x = np.random.normal(mu, sigma, 40)

count, bins, ignored = plt.hist(x, 30, normed=True)
plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
         np.exp(-(bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=2, color='r')
plt.show()

test_stat, p_val = stats.ttest_1samp(x, 120)
p_val /= 2
print("test statistic: ", round(test_stat, 2), "p-value: ", round(p_val, 4))


# 1.2 Perform chi-square test for variance

def chi2test_var(x, sigma_square, tail):
    test_stat = (len(x) - 1) * np.var(x) / sigma_square
    if tail == 'u':
        p_val = 1 - stats.chi2.cdf(test_stat, len(x) - 1)
    elif tail == 'l':
        p_val = stats.chi2.cdf(test_stat, len(x) - 1)
    else:
        p_val = 2 * (1 - stats.chi2.cdf(test_stat, len(x) - 1))
    return test_stat, p_val

test_stat, p_val = chi2test_var(x, 20**2, 'l')
print("test statistic: ", round(test_stat, 2), "p-value: ", round(p_val, 4))


# 2.1 Test for equal variance

np.random.seed(1234)
mu1 = 130
sigma1 = 15
x1 = np.random.normal(mu1, sigma1, 50)

mu2 = 150
sigma2 = 16
x2 = np.random.normal(mu2, sigma2, 30)

test_stat, p_val = stats.levene(x1, x2)
print("test statistic: ", round(test_stat, 2), "p-value: ", round(p_val, 4))


# 2.2 Test for difference
test_stat, p_val = stats.ttest_ind(x1, x2,  equal_var=True)
print("test statistic: ", round(test_stat, 2), "p-value: ", round(p_val, 4))


# 3.1 Paired-sample t Test
np.random.seed(1234)
mu1 = 150
sigma1 = 16
x1 = np.random.normal(mu1, sigma1, 30)
x2 = x1 + np.random.normal(10, 15, 30)

test_stat, p_val = stats.ttest_rel(x1, x2)
print("test statistic: ", round(test_stat, 2), "p-value: ", round(p_val, 4))


# 4.1 One-way ANOVA

x = np.random.normal(0.60, 0.01, 100).tolist()
x.extend(np.random.normal(0.65, 0.12, 100).tolist())
x.extend(np.random.normal(0.72, 0.15, 100).tolist())
x.extend(np.random.normal(0.80, 0.15, 100).tolist())
x.extend(np.random.normal(0.90, 0.05, 100).tolist())

algorithm = np.repeat("SVM", 100).tolist()
algorithm.extend(np.repeat("LR", 100).tolist())
algorithm.extend(np.repeat("RF", 100).tolist())
algorithm.extend(np.repeat("DNN", 100).tolist())
algorithm.extend(np.repeat("IAN", 100).tolist())

tukey = pairwise_tukeyhsd(endog=np.array(x), groups=np.array(algorithm), alpha=0.05)
print(tukey.summary())
tukey.plot_simultaneous()
