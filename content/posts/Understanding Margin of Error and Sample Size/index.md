---
title: "Understanding Margin of Error and Sample Size"
author: "Will Barker"
date: 2020-11-21
tags: ["stats"]
---

Part 1 of 3 on a series of notes covering margin of error, power, and sample size calculations. 

Notes, with questions and examples, taken from the following reading: https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Power/BS704_Power_print.html

<!--more--> 

## Confidence Interals, Margins of Error, and Sample Sizes

Confidence intervals take the following general form: Point Estimate $\pm$ Margin of Error<br><br>
For confidence intervals based on normal data, this looks like:

$$ \bar{X} \pm E $$

- $\bar{X}$ is the sample mean generated through the experiment (our point estimate)
- $E$ is the margin of error, calculated as:
$$ E=Z\frac{\sigma}{\sqrt{n}} $$<br>
- $Z$ is the Z statistic of a standard normal distribution for a desired confidence level (Z = 1.96 for 95% confidence)
- $\sigma$ is the standard deviation of the population $\mu$ (as best as we know/can estimate it)
- $\sqrt{n}$ is the square root of the sample size

In planning experiments we need to determine the sample size required to achieve a sufficiently small margin of error. If the margin of error is too wide then the test is fairly uninformative. To determine the sample size needed first we need to define the desired margin of error, and then we can use algebra to solve:

$$ E = Z\frac{\sigma}{\sqrt{n}} $$<br>
$$ \sqrt{n}E = Z\sigma $$<br>
$$ \sqrt{n} = \frac{Z\sigma}{E} $$<br>
$$ n = \left(\frac{Z\sigma}{E}\right)^2 $$

$\sigma$ can be difficult to estimate at the outset of a experiment, so it can be appropriate to use a value for the standard deviation from a previous study done to a comparable population. However it's determined, $\sigma$ should be a conservative estimate (i.e. as large as is reasonable) so that the resulting sample size isn't too small.

The following examples demonstrate these sample size calculations for different scenarios and random variables.

### Sample Size for One Sample, Continuous Outcome
Example 1:
- *An investigator wants to estimate the mean systolic blood pressure in children with congenital heart disease who are between the ages of 3 and 5. How many children should be enrolled in the study? The investigator plans on using a 95% confidence interval (so Z=1.96) and wants a margin of error of 5 units. The standard deviation of systolic blood pressure is unknown, but the investigators conduct a literature search and find that the standard deviation of systolic blood pressures in children with other cardiac defects is between 15 and 20.*


```python
from scipy.stats import norm
```


```python
Z = norm.ppf(0.975)
std = 20
E = 5
```


```python
n = ((Z*std)/E)**2
```


```python
n
```




    61.46334113110599



In order to ensure a 95% confidence interval the study will need 62 participants (rounding up). Selecting a smaller sample size could potentially produce a confidence interval with a larger margin of error.

Question 1:
- *An investigator wants to estimate the mean birth weight of infants born full term (approximately 40 weeks gestation) to mothers who are 19 years of age and under. The mean birth weight of infants born full-term to mothers 20 years of age and older is 3,510 grams with a standard deviation of 385 grams. How many women 19 years of age and under must be enrolled in the study to ensure that a 95% confidence interval estimate of the mean birth weight of their infants has a margin of error not exceeding 100 grams?*
- *If 5% of women are expected to deliver prematurely, how many participants should there be to account for this possibility?*


```python
Z = norm.ppf(0.975)
std = 385
E = 100
```


```python
n = ((Z*std)/E)**2
```


```python
n
```




    56.94002336973868



In order to ensure a 95% confidence interval the study will need 57 participants. If 5% of women are expected to deliver prematurely then we would need $\frac{n}{0.95} = 60$ participants


```python
expected_premature = 0.05
n = (n/(1 - expected_premature))
```


```python
n
```




    59.93686670498809



### Sample Size for One Sample, Binary Outcome (Bernoulli)

In experiments to estimate the proportion of successes in a variable with a binary outcome (yes/no, AKA a bernoulli random variable), the formula becomes:

$$ n = p(1-p)\left(\frac{Z}{E}\right)^2 $$


- $n$ is equal to the variance of a bernoulli trial multiplied by the square of the desired confidence Z score over the margin of error

Working backwards to get the margin of error:

$$ n = p(1-p)\left(\frac{Z}{E}\right)^2 = \sigma^2\left(\frac{Z}{E}\right)^2 $$<br>
$$ \frac{n}{\sigma^2} = \left(\frac{Z}{E}\right)^2 $$<br>
$$ \sqrt{\frac{n}{\sigma^2}} = \frac{Z}{E} $$<br>
$$ \frac{\sqrt{n}}{\sigma} = \frac{Z}{E} $$<br>
$$ E\frac{\sqrt{n}}{\sigma} = Z $$<br>
$$ \frac{E}{\sigma} = \frac{Z}{\sqrt{n}} $$<br>
$$ E = Z\frac{\sigma}{\sqrt{n}} $$

In planning an experiment, $p$ is our estimate of the propensity for the binary outcome to be a success and $1-p$ is the propensity for it to be failure. If no knowledge is known for an estimate of $p$, using 0.5 (50/50 chance) will maximize the variance and the sample size.

Example 2:
- *An investigator wants to estimate the proportion of freshmen at his University who currently smoke cigarettes (i.e., the prevalence of smoking). How many freshmen should be involved in the study to ensure that a 95% confidence interval estimate of the proportion of freshmen who smoke is within 5% of the true proportion?*


Since we have no information of the proportion of freshmen who smoke, we use 0.5 to estimate the sample size as follows:


```python
p = 0.5
Z = norm.ppf(0.975)
E = 0.05
```


```python
n = p*(1-p)*(Z/E)**2
```


```python
n
```




    384.14588206941244



To ensure a 95% confidence interval estimate of the proportion of freshmen who smoke is within 5% of the true population, a sample size of 385 is needed.

Question 2:
- *Suppose that a similar study was conducted 2 years ago and found that the prevalence of smoking was 27% among freshmen. If the investigator believes that this is a reasonable estimate of prevalence 2 years later, it can be used to plan the next study. Using this estimate of p, what sample size is needed (assuming that again a 95% confidence interval will be used and we want the same level of precision)?*


```python
p = 0.27
Z = norm.ppf(0.975)
E = 0.05
```


```python
n = p*(1-p)*(Z/E)**2
```


```python
n
```




    302.86061342352474



To ensure a 95% confidence interval estimate of the proportion of freshmen who smoke is within 5% of the true population, a sample size of 303 is needed.


Example 3:
- *An investigator wants to estimate the prevalence of breast cancer among women who are between 40 and 45 years of age living in Boston. How many women must be involved in the study to ensure that the estimate is precise? National data suggest that 1 in 235 women are diagnosed with breast cancer by age 40. This translates to a proportion of 0.0043 (0.43%) or a prevalence of 43 per 10,000 women. Suppose the investigator wants the estimate to be within 10 per 10,000 women with 95% confidence.*


```python
p = 43/10000
Z = norm.ppf(0.975)
E = 10/10000
```


```python
n = p*(1-p)*(Z/E)**2
```


```python
n
```




    16447.244355390107



A sample size of n=16447 will ensure a 95% confidence interval estimate of the prevelance of breast cancer is within 0.10 (10 women per 10,000).

- Suppose this sample size isn't feasible, and the investigators thought a sample size of 5,000 would be practical
- How precisely can we estimate the prevalence with a sample size of n=5,000?

The confidence interval formula to estimate prevalence is:<br><br>
$$ \hat{p}\pm Z\sqrt{\frac{\hat{p}(1-\hat{p})}{N}} $$

This is just the sample mean plus/minus the Z score multiplied the standard error of the mean. If we assume the prevalence of breast cancer in the sample will be close to that based on national data, we can expect the margin of error to be approximately:

$$ Z\sqrt{\frac{\hat{p}(1-\hat{p})}{N}} = 1.96\sqrt{\frac{0.0043(1-0.00.43)}{5000}} = 0.0018 $$


```python
sample_size = 5000
```


```python
E = Z*((p*(1-p))/sample_size)**(1/2)
```


```python
E
```




    0.0018136837847535663



With n=5,000 women in the sample, a 95% confidence interval would be expected to have a margin of error of 0.0018 (18 per 10,000). The investigators would need to decide if this is precise enough to answer the question. This comes with the assumption that the propensity for one to get breast cancer in Boston is similar to the propensity to get it nationally, which might be a stretch.

### Sample Sizes for Two Independent Samples, Continuous Outcome

For studies where the plan is to estimate the difference in means between two independent populations, the formula for determining sample sizes becomes:

$$ n_i = 2\left(\frac{Z\sigma}{ES_p}\right)^2 $$

- $n_i$ is the sample size required in each group
- $Z$ is again the Z score from the standard normal distribution for the confidence level used
- $E$ is the desired margin of error
- $\sigma$ is the standard deviation of the outcome variable
- $S_p$ is the pooled estimate of the common standard deviation between the two populations, calculated as:

$$ S_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{(n_1 + n_2 - 2)}} $$

If data is available on variability of the outcome in each population, then $S_p$ can be computed and used in the sample size formula. Usually though there's only data on the variance in one group, usually the control.


When planning an investigation data is often available from other trials that involved a placebo or control group, and a standard deviation from these trials can be used for the experimental (non-control) group in this trial. When this is the case we forget about $S_p$ and just use the following:

$$ n_i = 2\left(\frac{Z\sigma}{E}\right)^2 $$

Note: sample size formula generates estimates for samples of equal size, alternative formulas can be used for samples of different sizes

Skipping Example 4 and going straight to Example 5:
- *An investigator wants to compare two diet programs in children who are obese. One diet is a low fat diet, and the other is a low carbohydrate diet. The plan is to enroll children and weigh them at the start of the study. Each child will then be randomly assigned to either the low fat or the low carbohydrate diet. Each child will follow the assigned diet for 8 weeks, at which time they will again be weighed. The number of pounds lost will be computed for each child. Based on data reported from diet trials in adults, the investigator expects that 20% of all children will not complete the study. A 95% confidence interval will be estimated to quantify the difference in weight lost between the two diets and the investigator would like the margin of error to be no more than 3 pounds. How many children should be recruited into the study?*
- *Again, the issue is determining the variability in the outcome of interest (σ), here the standard deviation in pounds lost over 8 weeks. To plan this study, investigators use data from a published study in adults. Suppose one such study compared the same diets in adults and involved 100 participants in each diet group. The study reported a standard deviation in weight lost over 8 weeks on a low fat diet of 8.4 pounds and a standard deviation in weight lost over 8 weeks on a low carbohydrate diet of 7.7 pounds.*

Can use the information in the second bullet to compute the pooled estimate of the standard deviation between low fat and low carbohydrate groups:


```python
n_fat = 100
n_carb = 100

std_fat = 8.4
std_carb = 7.7

std_pooled = (((n_fat - 1)*std_fat**2 \
               + (n_carb - 1)*std_carb**2) \
               / (n_fat + n_carb - 2))**(1/2)
```


```python
std_pooled
```




    8.057605103254938



$S_p = 8.1 $, rounding up. We will use this as $\sigma$ in our experiment, and do not need to multiply the margin of error by the pooled variance (drop $ES_p$ in the denominator and just use $E$):


```python
Z = norm.ppf(0.975)
E = 3

n = 2*((Z*std_pooled)/E)**2
```


```python
n
```




    55.423714207459135



$n = 56$, rounding up. This means that $2 \times n_i = 2 \times 56 = 112$ children should be recruited for the study (not counting attrition). If we factor in an attrition rate of 20%:


```python
attrition_rate = 0.2
n_total = (n * 2) / (1 - attrition_rate)
```


```python
n_total
```




    138.55928551864784



Factoring in attrition, about 140 children should participate in the study.

### Sample Size for Matched Samples, Continuous Outcomes

In studies where the plan is to estimate the mean difference of a continuous outcome based on matched (i.e. paired) data:

$$ n = \left(\frac{Z\sigma_d}{E}\right)^2 $$<br>

In this case, $\sigma_d$ is the standard deviation of the difference scores. The standard deviation between the paired data points must be used here, you can't estimate the difference using past trials.

### Sample Sizes for Two Independent Samples, Binary Outcome

In studies where the plan is to estimate the difference in proportions between two independent populations, the formula for determining the sample sizes required in each comparison group is:

$$ n_i = \{p_1(1-p_1) + p_2(1-p_2)\}\left(\frac{Z}{E}\right)^2 $$

- $n_i$ is the sample size required in each group
- $\{p_1(1-p_1) + p_2(1-p_2)\}$ is their pooled variance
- $Z$ is again the Z score from the standard normal distribution for the confidence level used
- $E$ is the desired margin of error
- $p_1$ and $p_2$ are the propensities for success in each group


To estimate the sample size we need to approximate $p_1$ and $p_2$, or if we have no prior intuitions and just want to generate the most conservative and largest sample sizes we can again use just 0.5.


If we're comparing an unknown group with a group that we know already have data on (e.g. the control group), we can use its proportion for both $p_1$ and $p_2$. Alternative formulas can be used with groups with different sample sizes

Example 6
- *An investigator wants to estimate the impact of smoking during pregnancy on premature delivery. Normal pregnancies last approximately 40 weeks and premature deliveries are those that occur before 37 weeks. The 2005 National Vital Statistics report indicates that approximately 12% of infants are born prematurely in the United States.5 The investigator plans to collect data through medical record review and to generate a 95% confidence interval for the difference in proportions of infants born prematurely to women who smoked during pregnancy as compared to those who did not. How many women should be enrolled in the study to ensure that the 95% confidence interval for the difference in proportions has a margin of error of no more than 4%?*


- *The sample sizes (i.e., numbers of women who smoked and did not smoke during pregnancy) can be computed using the formula shown above. National data suggest that 12% of infants are born prematurely.*


```python
# using the proportion of p=0.12 for both groups (smoking and non-smoking)

Z = norm.ppf(0.975)
E = 0.04
p = 0.12

n = (p*(1-p) + p*(1-p))*(Z/E)**2
```


```python
n
```




    507.07256433162445



A sample size of $n_1=508$ women who smoked during pregnancy and $n_2=508$ who did not during pregnancy will ensure that the 95% confidence interval for the difference in proportions who deliver prematurely will have a margin of error of no more than 4%.


Attrition could be a factor in this trial as confounding factors could happen to either group (someone stops/starts smoking or decides to drop out for whatever reason)

### Note to be continued in Part 2: Understanding Statistical Power