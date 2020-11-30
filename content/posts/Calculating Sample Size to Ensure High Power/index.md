---
title: "Calculating Sample Size to Ensure High Power"
author: "Will Barker"
date: 2020-11-29
tags: ["stats"]
---

Part 3 of 3 on a series of notes covering margin of error, power, and sample size calculations. 

Notes, with questions and examples, taken from the following reading: https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Power/BS704_Power_print.html

<!--more--> 

## Ensuring a Test has High Power

Power values of 80-90% are commonly accepted as norms when running hypothesis tests. Certain power levels can guaranteed in studies by including enough samples to control the variability in the parameter of interest.

The input for the sample size formulas include desired power, the level of significance, and the effect size. Effect size is selected to represent a meaningful or practically important difference in the parameter of interest.

The formulas below produce minimum sample sizes to ensure that their associated flavours of hypothesis tests will have a specified probability of rejecting the null hypothesis when it is false (i.e. power). Additionally, for certain studies one may need to factor in the likelihood of attrition or lose in the samples afterwards.

### Sample Size for One Sample, Continuous Outcome

In a hypothesis test comparing the mean of a continuous normal variable in a single population to a known mean, the hypotheses are:

$$ H_0: \mu = \mu_0 $$
$$ H_A: \mu \neq \mu_0 $$

Where $\mu_0$ is the known mean (e.g. historical control).

The formula for determining sample size to ensure the test has a specified power is:

$$ n = \left(\frac{Z_{1-\alpha/2} + Z_{1-\beta}}{ES}\right)^2 $$

- $\alpha$ is the selected level of significance and $Z_{1-\alpha/2}$ is the value from the standard normal distribution holding $1-\alpha/2$ below it
- $1-\beta$ is the selected power, and $Z_{1-\beta}$ is the value from the standard normal distribution holding $1-\beta$ below it

For 80% power, this associated Z value is $Z_{0.80} = 0.84$. For 90% power, it is $Z_{0.90} = 1.282$.


```python
from scipy.stats import norm
import numpy as np
```


```python
norm.ppf(0.8)
```




    0.8416212335729143




```python
norm.ppf(0.9)
```




    1.2815515655446004



$ES$ is the effect size, defined as follows:

$$ ES = \frac{\lvert\mu_1 - \mu_0\rvert}{\sigma} $$

- $\mu_0$ is the mean under $H_0$
- $\mu_1$ is the mean under $H_1$
- $\sigma$ is the standard deviation of the outcome of interest


The numerator of the effect size is the absolute difference in means, $\lvert\mu_1 - \mu_0\rvert$, representing what is considered a meaningful or important difference in the population.

It can be difficult to underestimate $\sigma$ at the outset of a test - in sample size calculations it is common to use a value from a previous study or a study performed on a comparable population. Regardless, $\sigma$ should always be conservative (i.e. reasonably large), so that the resultant sample size isn't too small.

### Example

- *An investigator hypothesizes that in people free of diabetes, fasting blood glucose, a risk factor for coronary heart disease, is higher in those who drink at least 2 cups of coffee per day. A cross-sectional study is planned to assess the mean fasting blood glucose levels in people who drink at least two cups of coffee per day. The mean fasting blood glucose level in people free of diabetes is reported as 95.0 mg/dL with a standard deviation of 9.8 mg/dL.7 If the mean blood glucose level in people who drink at least 2 cups of coffee per day is 100 mg/dL, this would be important clinically. How many patients should be enrolled in the study to ensure that the power of the test is 80% to detect this difference? A two sided test will be used with a 5% level of significance.*


```python
two_cups_glucose = 100.0
mean_glucose = 95.0
std_glucose = 9.8

effect_size = (two_cups_glucose - mean_glucose)/std_glucose
effect_size
```




    0.5102040816326531



The effect size represents a meaningful standardized difference in the population mean - 95 mg/dL vs. 100 mg/dL, or 0.51 standard deviation units different.


```python
Z_significance = norm.ppf(1 - (0.05/2))
beta = 0.2
Z_power = norm.ppf(1 - beta)

n_patients = ((Z_significance + Z_power)/effect_size)**2
n_patients
```




    30.152256387475454



Therefore a sample size of n=31 (rounding up) will ensure that a two-sided test with $\alpha = 0.05$ has 80% power to detect a 5 mg/dL difference in mean fasting blood glucose levels.

- *In the planned study, participants will be asked to fast overnight and to provide a blood sample for analysis of glucose levels. Based on prior experience, the investigators hypothesize that 10% of the participants will fail to fast or will refuse to follow the study protocol.*

Factoring in 10% attritition to hit the needed 31 participants:


```python
31 / (1 - 0.1)
```




    34.44444444444444



Factoring in an attrition rate of 10%, 35 participants should be enrolled in the study.

### Sample Size for One Sample, Bernoulli Outcome

In studies where the plan is to perform a hypothesis test comparing the proportion of successes in a bernoulli variable in a single population to a known proportion, the hypotheses become:

$$ H_0: p = p_0 $$
$$ H_A: p \neq p_0 $$

The formula for calculating sample size remains the same as the one for one sample, continuous outcome. This is because a bernoulli random variable approximates to a normal distribution across many trials due to CLT:

$$ n = \left(\frac{Z_{1-\alpha/2} + Z_{1-\beta}}{ES}\right)^2 $$

The effect size $ES$ is calculated as:

$$ ES = \frac{\lvert p_A - p_0\rvert}{\sqrt{p_0(1-p_0)}} $$

- where $p_0$ is the proportion under $H_0$ and $p_A$ is the proportion under $H_A$
- the numerator is again a meaningful difference in proportions

We use $p_0$ for the standard deviation calculation in the denominator because we want to measure the effect size of $p_A$, the proportion in our alternate hypothesis, in relation to what we already know/have observed about the population.

### Example

- *A medical device manufacturer produces implantable stents. During the manufacturing process, approximately 10% of the stents are deemed to be defective. The manufacturer wants to test whether the proportion of defective stents is more than 10%. If the process produces more than 15% defective stents, then corrective action must be taken. Therefore, the manufacturer wants the test to have 90% power to detect a difference in proportions of this magnitude. How many stents must be evaluated? For you computations, use a two-sided test with a 5% level of significance.*


```python
p0_stents = 0.1
pA_stents = 0.15

effect_size = np.abs(pA_stents - p0_stents)/\
              np.sqrt(p0_stents * (1 - p0_stents))

effect_size
```




    0.1666666666666666



We could round this effect size up to 0.17 - doing so would simplify our understanding of what the standardized difference we're looking for is, but it would lower the number of samples in our test by way of increasing the denominator.


```python
alpha = 0.05
beta = 0.1

Z_significance = norm.ppf(1 - (alpha/2))
Z_power = norm.ppf(1 - beta)

n_stents = np.square((Z_significance + Z_power)/effect_size)
n_stents
```




    378.26723021186257



Therefore, 379 stents should be evaluated to ensure that a two-sided test with $\alpha = 0.05$ and 90% power would detect a 5% difference (the delta between a 10% and 15% defective rate) in the proportion of defective stents produced.

### Sample Sizes for Two Independent Samples, Continuous Outcomes

When the plan is to perform a hypothesis test on the mean difference of a continuous random variable (CRV) in two independent populations, the hypotheses of interest are:

$$ H_0: \mu_1 = \mu_2 $$
$$ H_A: \mu_1 \neq \mu_2 $$

where $\mu_1$ and $\mu_2$ are the means in the two comparison populations.

The formulas for determining sample size and effect size:

$$ n_i = 2\left(\frac{Z_{1-\alpha/2} + Z_{1-\beta}}{ES}\right)^2 $$

$$ ES = \frac{\lvert\mu_1 - \mu_0\rvert}{\sigma} $$

Where $n_i$ is the sample size required in each group (i=1,2). When doing a hypothesis test on two independent groups, the pooled estimate of standard deviation $S_p$ is used:

$$ S_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{(n_1 + n_2 - 2)}} $$

If the variability of each of the two populations is known at the outset of the test, then we can use algebra to get the sample sizes by reversing the equation (and setting $n_1 = n_2$, generating samples of equal size). However, it is usually the case that data on the variability of the outcome will only be available for a single population that you're testing an alternative against. This can be used as a substitute for the standard deviation in the effect size to plan the test.

### Example

- *An investigator is planning a study to assess the association between alcohol consumption and grade point average among college seniors. The plan is to categorize students as heavy drinkers or not using 5 or more drinks on a typical drinking day as the criterion for heavy drinking. Mean grade point averages will be compared between students classified as heavy drinkers versus not using a two independent samples test of means. The standard deviation in grade point averages is assumed to be 0.42 and a meaningful difference in grade point averages (relative to drinking status) is 0.25 units. How many college seniors should be enrolled in the study to ensure that the power of the test is 80% to detect a 0.25 unit difference in mean grade point averages? Use a two-sided test with a 5% level of significance.*


```python
gpa_std = 0.42
gpa_delta = 0.25

alpha = 0.05
beta = 0.2
```

Since variability is only known for average GPA (not for the two populaitons, heavy drinkers vs. non heavy drinkers), we'll use it to plan the study.


```python
effect_size = gpa_delta/gpa_std
effect_size
```




    0.5952380952380952



Pretty large effect size (i.e. we're testing for a pretty obvious difference between the two populations), so we won't need as many samples to achieve our desired power.


```python
Z_significance = norm.ppf(1 - (alpha/2))
Z_power = norm.ppf(1 - beta)

n_students = 2 * np.square((Z_significance + Z_power)/effect_size)
n_students
```




    44.30535632445374



Therefore in each group we would need 44 students - 44 heavy drinkers, 44 who aren't heavy drinkers, 88 students total.

### Sample Size for Matched Samples, Continuous Outcome

- in studies where the plan is to perform a hypothesis test on the mean difference in a continuous outcome variable based on matched data:

$$ H_0: \mu_d = 0 $$
$$ H_A: \mu_d \neq 0 $$

Where $\mu_d$ is the mean difference in the population.

The formula for sample size is again:

$$ n_i = \left(\frac{Z_{1-\alpha/2} + Z_{1-\beta}}{ES}\right)^2 $$

While effect size is calculated as:

$$ ES = \frac{\mu_d}{\sigma_d} $$

Where $\sigma_d$ is the standard deviation of the difference in the outcome (i.e. difference based on measurements over time/between matched pairs).

### Example

- *An investigator wants to evaluate the efficacy of an acupuncture treatment for reducing pain in patients with chronic migraine headaches. The plan is to enroll patients who suffer from migraine headaches. Each will be asked to rate the severity of the pain they experience with their next migraine before any treatment is administered. Pain will be recorded on a scale of 1-100 with higher scores indicative of more severe pain. Each patient will then undergo the acupuncture treatment. On their next migraine (post-treatment), each patient will again be asked to rate the severity of the pain. The difference in pain will be computed for each patient. A two sided test of hypothesis will be conducted, at α =0.05, to assess whether there is a statistically significant difference in pain scores before and after treatment. How many patients should be involved in the study to ensure that the test has 80% power to detect a difference of 10 units on the pain scale? Assume that the standard deviation in the difference scores is approximately 20 units.*


```python
pain_std = 20
pain_delta = 10
alpha = 0.05
beta = 0.2

effect_size = pain_delta/pain_std
effect_size
```




    0.5




```python
Z_significance = norm.ppf(1 - (alpha/2))
Z_power = norm.ppf(1 - beta)

n_samples = np.square((Z_significance + Z_power) / effect_size)
n_samples
```




    31.395518937396353



Therefore a sample size n=32 patients with migraines will ensure that a two sided test with $\alpha=0.05$ has 80% power to detect a mean difference of 10% pain before and after the treatment, assuming all patients complete the treatment.

### Sample Sizes for Two Independent Samples, Dichotomous Outcomes

In studies where the plan is to perform a hypothesis test comparing the proportions of successes in two independent populations, the hypotheses of interest are:

$$ H_0: p_1 = p_2 $$
$$ H_A: p_1 \neq p_2 $$

Where $p_1$ and $p_2$ are the proportions in the two comparison populations.

The formulas for determining sample size and effect size are:

$$ n_i = 2\left(\frac{Z_{1-\alpha/2} + Z_{1-\beta}}{ES}\right)^2 $$<br>
$$ ES = \frac{\lvert{p_1 - p_2}\rvert}{\sqrt{p(1-p)}} $$

- $n_i$ is the sample size required for each group (i=1,2).
- $\lvert{p_1 - p_2}\rvert$ is the absolute value of the difference in proportions between the two groups expected under the alternate hypothesis $H_A$.
- $p$ is the overall proportion, based on pooling the data from the two comparison groups (can be computed by taking the mean of the proportions in the two groups, assuming that the groups will be of approximately equal size.

### Example

- *Clostridium difficile (also referred to as "C. difficile" or "C. diff.") is a bacterial species that can be found in the colon of humans, although its numbers are kept in check by other normal flora in the colon. Antibiotic therapy sometimes diminishes the normal flora in the colon to the point that C. difficile flourishes and causes infection with symptoms ranging from diarrhea to life-threatening inflammation of the colon. Illness from C. difficile most commonly affects older adults in hospitals or in long term care facilities and typically occurs after use of antibiotic medications.*<br><br>

- *In recent years, C. difficile infections have become more frequent, more severe and more difficult to treat. Ironically, C. difficile is first treated by discontinuing antibiotics, if they are still being prescribed. If that is unsuccessful, the infection has been treated by switching to another antibiotic. However, treatment with another antibiotic frequently does not cure the C. difficile infection. There have been sporadic reports of successful treatment by infusing feces from healthy donors into the duodenum of patients suffering from C. difficile. (Yuk!) This re-establishes the normal microbiota in the colon, and counteracts the overgrowth of C. diff.*<br><br>

- *The efficacy of this approach was tested in a randomized clinical trial reported in the New England Journal of Medicine (Jan. 2013). The investigators planned to randomly assign patients with recurrent C. difficile infection to either antibiotic therapy or to duodenal infusion of donor feces. In order to estimate the sample size that would be needed, the investigators assumed that the feces infusion would be successful 90% of the time, and antibiotic therapy would be successful in 60% of cases. How many subjects will be needed in each group to ensure that the power of the study is 80% with a level of significance α = 0.05?*


```python
p_feces = 0.9
p_anti =  0.6
p = np.mean([p_feces, p_anti])

effect_size = np.abs(p_feces - p_anti)/np.sqrt(p * (1 - p))
effect_size
```




    0.692820323027551



Again, pretty sizeable effect size.


```python
alpha = 0.05
beta = 0.2

Z_significance = norm.ppf(1 - alpha/2)
Z_power = norm.ppf(1 - beta)

n_subjects = 2 * np.square((Z_significance + Z_power) / effect_size)
n_subjects
```




    32.70366555978786



Each group would need about 33 subjects, so (66 subjects total) to detect a 30% difference in the two methods with $\alpha = 0.05$ and 80% power.

### Concluding Notes

Determining the appropriate design of a study is more important than the analysis; you can always re-analzye the data, you can't always just redo studies. We need a sample size large enough to answer the research question, byachieving acceptable margins of error and powers for the results.


```python

```
