---
title: "Randomized Experiments/RCTs"
author: "Will Barker"
date: 2021-03-20
tags: ["stats", "causal-inference", "experimentation"]
---

Notes from <i>Causal Inference for the Brave and True</i>

https://matheusfacure.github.io/python-causality-handbook/02-Randomised-Experiments.html

<!--more--> 

---

# Randomized Experiments

## The Golden Standard

Association becomes causation when there is no bias between treatment & control groups.
- There's no difference between them except for the treatment itself
- The outcome of the untreated group is the same as the counterfactual of the treated group
- $E[Y_{0} \vert T = 0] = E[Y_{0} \vert T = 1]$

Randomized Experiments, otherwise known as Randomized Controlled Trials (RCTs), can make bias vanish:
- Randomly assigning individuals in a population to either a treatment or control group
- Doesn't need to be 50/50 split, as long as the sample size is large enough to be representative
- Randomization annihilates bias by making the potential outcomes independent of the treatment


$(Y_{0}, Y_{1}) \perp\!\!\!\perp T$
- Where $\perp\!\!\!\perp$ is the symbol for conditional independence
- This means that the potential outcomes are independent of the treatment
- Emphasis on potential outcomes $Y_{0}$ or $Y_{1}$
- In randomized trials we don't want the outcome $Y$ to be independent of the treatment, because we think the treatment causes the outcome
- But saying the potential outcomes $Y_{0}$ or $Y_{1}$ are independent of the treatment is to say that in expectation, they are the same between the control and treatment groups (i.e. the groups are comparable)

$(Y_{0}, Y_{1}) \perp T$
- Where $\perp$ essentially means dependence
- This means that the treatment is the only thing generating a difference between the outcome in the treated and control groups
- Which, if this is the case, implies $E[Y_{0} \vert T = 0] = E[Y_{0} \vert T = 1] = E[Y_{0}]$
- Which gives us $E[Y \vert T = 1] - E[Y \vert T = 0] = E[Y_{1} - Y_{0}] = ATE$
- Meaning the randomization allows us to just use the simple difference in means between treatment and control as the treatment effect

## In a School Far, Far Away

- Let's say we wanted to know if remote learning has a positive or negative impact on student performance
- If we were to compare students in schools that give mostly online classes to those that just use traditional classrooms, we would run the risk of mistaking association for causation (bias exists)
- $T = 1$ for online schools and $T = 0$ for traditional schools

Potential biases:
- Online schools attract more studious, disciplined students $\rightarrow E[Y_{0} \vert T = 1] > E[Y_{0} \vert T = 0]$ (positive bias)
- Online schools consist of poorer students who cannot afford traditional schooling $\rightarrow E[Y_{0} \vert T = 1] < E[Y_{0} \vert T = 0]$ (negative bias)
- We could still speak to correlation, but can't make any convincing claims about causality

Randomly assigning the online and traditional classes to students solves this
- On average the treatment is the only difference between the two groups


```python
import pandas as pd
import numpy as np
```


```python
data = pd.read_stata("113462-V1/data-file-and-program/ReplicationData2.dta")
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gpa</th>
      <th>cum_total_crds</th>
      <th>gender</th>
      <th>asian</th>
      <th>black</th>
      <th>hawaiian</th>
      <th>hispanic</th>
      <th>unknown</th>
      <th>white</th>
      <th>ethnic_dummy</th>
      <th>format_ol</th>
      <th>format_blended</th>
      <th>sat_math_NEW</th>
      <th>sat_verbal_NEW</th>
      <th>enroll_count</th>
      <th>format_f2f_v_ol</th>
      <th>format_f2f_v_blended</th>
      <th>format_combined_v_f2f</th>
      <th>falsexam</th>
      <th>experiment1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.014</td>
      <td>63.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>540.0</td>
      <td>540.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>3.720</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>590.0</td>
      <td>630.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>650.0</td>
      <td>570.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>690.0</td>
      <td>690.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>480.0</td>
      <td>420.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_groups = data.assign(class_format = np.select([data["format_ol"].astype(bool), data["format_blended"].astype(bool)],
                                                   ["online", "blended"],
                                                   default="face_to_face")).groupby(["class_format"]).mean()
```


```python
data_groups
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gpa</th>
      <th>cum_total_crds</th>
      <th>gender</th>
      <th>asian</th>
      <th>black</th>
      <th>hawaiian</th>
      <th>hispanic</th>
      <th>unknown</th>
      <th>white</th>
      <th>ethnic_dummy</th>
      <th>format_ol</th>
      <th>format_blended</th>
      <th>sat_math_NEW</th>
      <th>sat_verbal_NEW</th>
      <th>enroll_count</th>
      <th>format_f2f_v_ol</th>
      <th>format_f2f_v_blended</th>
      <th>format_combined_v_f2f</th>
      <th>falsexam</th>
      <th>experiment1</th>
    </tr>
    <tr>
      <th>class_format</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>blended</td>
      <td>3.127560</td>
      <td>34.509934</td>
      <td>0.561404</td>
      <td>0.230088</td>
      <td>0.115044</td>
      <td>0.017699</td>
      <td>0.008850</td>
      <td>0.008850</td>
      <td>0.619469</td>
      <td>0.416667</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>620.413793</td>
      <td>579.554795</td>
      <td>2.421053</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>50.018696</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>face_to_face</td>
      <td>3.100013</td>
      <td>31.566413</td>
      <td>0.573620</td>
      <td>0.146154</td>
      <td>0.084615</td>
      <td>0.003846</td>
      <td>0.026923</td>
      <td>0.000000</td>
      <td>0.738462</td>
      <td>0.261538</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>625.521173</td>
      <td>590.618893</td>
      <td>2.525714</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>29.138031</td>
      <td>0.493289</td>
    </tr>
    <tr>
      <td>online</td>
      <td>3.061357</td>
      <td>36.774096</td>
      <td>0.526012</td>
      <td>0.220472</td>
      <td>0.055118</td>
      <td>0.007874</td>
      <td>0.023622</td>
      <td>0.023622</td>
      <td>0.669291</td>
      <td>0.330709</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>615.911950</td>
      <td>570.251572</td>
      <td>2.375723</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>40.963379</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We can use the difference in means between groups as the $ATE$:
- Looking at GPA, we can conclude that $ATE = E[Y_{1} - Y_{0}] = 3.06 - 3.10 = -0.04$ (comparing `online` and `face_to_face` groups)
- Online classes caused students to perform slightly worse
- Is this difference significant though, maybe another question

Either way the randomized experiment wiped out any bias between the groups:
- We can see that `gender` is fairly evenly distributed between groups
- `white` students/variable slightly overindexed in `face_to_face` however; these slight imbalances are due to small dataset size

## The Ideal Experiment

RCTs are the most reliable way to uncover causal effects - a well designed RCT is a scientist's dream.

Sometimes however, we can't control the assignment mechanism due to cost/ethical reasons:
- e.g. If we wanted to understand the effects of smoking during preganancy, we couldn't just force a random portion of moms to smoke while they were pregnant
- e.g. A big bank couldn't just give random lines of credit to customers to measure the impact on churn
- Conditional randomization can help lower the cost sometimes
- Nothing can be done for unethical/unfeasible experiments though
- Always worth it to ask what the ideal experiment would be, this can shed some light on how to uncover the causal effect without perfect conditions

## The Assignment Mechanism

Causal inference techniques try to identify the assignment mechanism of the treatments.
- In RCTs the assignment mechanism is pure randomness
- Understanding the assignment mechanism can make inference more certain

- Assignment mechanisms can't just be found looking at associations in the data through EDA
- In causal questions you can usually argue both ways: X causes Y, Y causes X, or Z causes X and Y, and the X/Y correlation is just spurious
- Understanding the assignment mechanism leads to more convincing answers and makes causal inference exciting

## Key Ideas

- RCTs make the treatment and control groups comparable; this is the equivalent to being able to see the counterfactuals for both groups
- When the potential outcome for the untreated $Y_{0}$ is the same for both the test and control groups, this allows us to call their difference in means for the outcome variable $Y$ as the average treatment effect $ATE$

Breaking this down from the original association equation:

$ATE = E[Y \vert T = 1] - E[Y \vert T = 0] = E[Y_{1} - Y_{0} \vert T = 1] + E[Y_{0} \vert T = 1] - E[Y_{0} \vert T = 0]$

Where the average treatment effect/difference in means/association between the groups is equal to the average treatment effect of the treated plus a bias term ($ATET + BIAS$)

RCTs make $Y_{0}$ the same between both groups so $E[Y_{0} \vert T = 1] - E[Y_{0} \vert T = 0] = x - x = 0$, eliminating the bias term

This reduces the average treatment effect to be equal to the average treatment effect of the treated:

$ATE = ATET \rightarrow E[Y \vert T = 1] - E[Y \vert T = 0] = E[Y_{1} - Y_{0} \vert T = 1]$

Long story short we can take the average treatment effect/difference in means between treatment to be equivalent to the causal effect of the treatment on the treatment group. Association becomes causation.

RCTs are great, but unfortunately they can't always be the solution due to ethical/cost/feasibility reasons.
